use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{rnn, Module, VarBuilder, RNN};

#[derive(Debug, Clone)]
pub struct BidirectionalLSTM {
    forward_lstm1: rnn::LSTM,
    backward_lstm1: rnn::LSTM,
    forward_lstm2: rnn::LSTM,
    backward_lstm2: rnn::LSTM,
    h0: Tensor,
    c0: Tensor,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
}

impl BidirectionalLSTM {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let h0 = vb.get((num_layers * 2, 1, hidden_size), "rnn_h0")?;
        let c0 = vb.get((num_layers * 2, 1, hidden_size), "rnn_c0")?;

        let lstm1_fw = rnn::lstm(
            input_size,
            hidden_size,
            rnn::LSTMConfig::default(),
            vb.pp("rnn"),
        )?;
        let lstm1_bw = rnn::lstm(
            input_size,
            hidden_size,
            rnn::LSTMConfig {
                direction: rnn::Direction::Backward,
                ..Default::default()
            },
            vb.pp("rnn"),
        )?;

        let lstm2_fw = rnn::lstm(
            2 * hidden_size,
            hidden_size,
            rnn::LSTMConfig {
                layer_idx: 1,
                ..Default::default()
            },
            vb.pp("rnn"),
        )?;
        let lstm2_bw = rnn::lstm(
            2 * hidden_size,
            hidden_size,
            rnn::LSTMConfig {
                layer_idx: 1,
                direction: rnn::Direction::Backward,
                ..Default::default()
            },
            vb.pp("rnn"),
        )?;

        Ok(Self {
            forward_lstm1: lstm1_fw,
            backward_lstm1: lstm1_bw,
            forward_lstm2: lstm2_fw,
            backward_lstm2: lstm2_bw,
            h0,
            c0,
            input_size,
            hidden_size,
            num_layers,
        })
    }

    fn apply_bidirectional_layer(
        &self,
        input: &Tensor,
        lstm_forward: &rnn::LSTM,
        lstm_backward: &rnn::LSTM,
        h0: &Tensor,
        c0: &Tensor,
        layer_idx: &i32,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (batch_size, seq_len, input_size) = input.dims3()?;

        // Print first and last 5 values of the original input
        let input_vec = input.to_vec3::<f32>()?;

        // Forward pass
        let h0_forward = h0.narrow(0, 0, 1)?.reshape((batch_size, h0.dim(2)?))?;
        let c0_forward = c0.narrow(0, 0, 1)?.reshape((batch_size, c0.dim(2)?))?;

        let state_forward = rnn::LSTMState {
            h: h0_forward.clone(),
            c: c0_forward.clone(),
        };

        let output_forward_states: Vec<rnn::LSTMState> =
            lstm_forward.seq_init(&input, &state_forward)?;
        let output_forward = Tensor::stack(
            &output_forward_states
                .iter()
                .map(|state| state.h().clone())
                .collect::<Vec<_>>(),
            1,
        )?;
        let last_forward_state = output_forward_states.last().unwrap().h().clone();

        // Backward pass
        let h0_backward = h0.narrow(0, 1, 1)?.reshape((batch_size, h0.dim(2)?))?;
        let c0_backward = c0.narrow(0, 1, 1)?.reshape((batch_size, c0.dim(2)?))?;

        let state_backward = rnn::LSTMState {
            h: h0_backward.clone(),
            c: c0_backward.clone(),
        };

        // Correctly reverse the input sequence
        let mut reversed_input = vec![vec![vec![0.0; input_size]; seq_len]; batch_size];
        for b in 0..batch_size {
            for t in 0..seq_len {
                for i in 0..input_size {
                    reversed_input[b][seq_len - t - 1][i] = input_vec[b][t][i];
                }
            }
        }
        let input_reversed = Tensor::new(reversed_input, input.device())?
            .to_dtype(DType::F32)?
            .reshape((batch_size, seq_len, input_size))?;

        // Print first and last 5 values of the reversed input
        // let reversed_input_vec = input_reversed.to_vec3::<f32>()?;

        let output_backward_states = lstm_backward.seq_init(&input_reversed, &state_backward)?;
        let output_backward = Tensor::stack(
            &output_backward_states
                .iter()
                .map(|state| state.h().clone())
                .collect::<Vec<_>>(),
            1,
        )?;

        // Use the last state of the backward LSTM (which corresponds to the first element of the original sequence)
        let last_backward_state = output_backward_states.last().unwrap().h().clone();

        // Combine the forward and backward hidden states for hn
        let hn = Tensor::cat(
            &[
                last_forward_state.unsqueeze(0)?,
                last_backward_state.unsqueeze(0)?,
            ],
            0,
        )?; // Shape: [2, 1, 128]
        let hn_concat = Tensor::cat(&[last_forward_state, last_backward_state], 1)?; // Shape: [1, 256]

        // Combine the forward and backwards cell states for cn
        let cn = Tensor::cat(
            &[
                output_forward_states.last().unwrap().c().clone(),
                output_backward_states.last().unwrap().c().clone(),
            ],
            0,
        )?; // Shape: [2, 1, 128]

        // The output_backward is already in the correct order for the original sequence
        let output = Tensor::cat(&[output_forward, output_backward], 2)?; // Shape: [1, 13, 256]

        Ok((output, (hn, cn)))
    }

    // New method that returns output and states
    pub fn forward_with_state(&self, xs: &Tensor) -> Result<(Tensor, (Tensor, Tensor))> {
        let (batch_size, seq_len, input_size) = xs.dims3()?;

        let h0 = &self
            .h0
            .expand((self.num_layers * 2, batch_size, self.hidden_size))?;
        let c0 = &self
            .c0
            .expand((self.num_layers * 2, batch_size, self.hidden_size))?;

        let h0_1 = h0.narrow(0, 0, 2)?;
        let h0_2 = h0.narrow(0, 2, 2)?;
        let c0_1 = c0.narrow(0, 0, 2)?;
        let c0_2 = c0.narrow(0, 2, 2)?;

        let (layer1_output, (hn1, cn1)) = self.apply_bidirectional_layer(
            xs,
            &self.forward_lstm1,
            &self.backward_lstm1,
            &h0_1,
            &c0_1,
            &1,
        )?;
        let (layer2_output, (hn2, cn2)) = self.apply_bidirectional_layer(
            &layer1_output,
            &self.forward_lstm2,
            &self.backward_lstm2,
            &h0_2,
            &c0_2,
            &2,
        )?;

        let final_hn = Tensor::cat(&[hn1, hn2], 0)?;
        let final_cn = Tensor::cat(&[cn1, cn2], 0)?;

        Ok((layer2_output, (final_hn, final_cn)))
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl Module for BidirectionalLSTM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_state(xs)?;
        Ok(output)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_bilstm() -> Result<()> {
        // Set dimensions
        let batch_size = 1;
        let seq_length = 5;
        let input_size = 3;
        let hidden_size = 4;
        let num_layers = 2;

        // Create a device (CPU in this case)
        let device = Device::Cpu;

        // Create the specific input tensor
        let x = Tensor::new(
            &[
                0.3241, -0.0404, -0.6861, 0.5437, -1.5626, 0.8695, 0.2145, -0.7496, -0.4951,
                1.3849, 0.7240, 0.1449, 2.2349, 0.0219, 0.3207,
            ],
            &device,
        )?
        .to_dtype(DType::F32)?
        .reshape((batch_size, seq_length, input_size))?;

        // Initialize the model
        // let vb = VarBuilder::zeros(DType::F32, &device);
        let model_path =
            "/home/singjc/Documents/github/alphapeptdeep/nbs_tests/model/simple_bilstm_example.pth";
        let vb = VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;
        let model: BidirectionalLSTM =
            BidirectionalLSTM::new(input_size, hidden_size, num_layers, &vb)?;

        // Create initial hidden and cell states
        let h0 = Tensor::zeros(
            (num_layers * 2, batch_size, hidden_size),
            DType::F32,
            &device,
        )?;
        let c0 = Tensor::zeros(
            (num_layers * 2, batch_size, hidden_size),
            DType::F32,
            &device,
        )?;

        // Forward pass
        let (output, (hn, cn)) = model.forward_with_state(&x)?;

        // Print results
        println!("Input shape: {:?}", x.shape());
        println!("Input:\n{}", x);

        println!("\nOutput shape: {:?}", output.shape());
        println!("Output:\n{}", output);

        println!("\nHidden state shape: {:?}", hn.shape());
        println!("Hidden state:\n{}", hn);

        println!("\nCell state shape: {:?}", cn.shape());
        println!("Cell state:\n{}", cn);

        // // Print model parameters
        // println!("\nModel Parameters:");
        // for (name, param) in model.named_parameters() {
        //     println!("{}:\n{}", name, param);
        // }

        Ok(())
    }

    #[test]
    fn test_bilstm_old() -> Result<()> {
        let device = Device::Cpu; // or Device::Cuda(0) for GPU
        let inp_sequence = [3f32, 1., 4., 1., 5., 9., 2.];
        let vb = VarBuilder::zeros(DType::F32, &device);

        // Create one lstm for the forward pass and one for the backward pass
        let lstm = candle_nn::lstm(2, 3, Default::default(), vb.pp("forward"))?;
        let lstm_rev = candle_nn::lstm(2, 3, Default::default(), vb.pp("backward"))?;

        // Apply the forward lstm and collect the results in states
        println!("Applying forward LSTM...");
        let mut states = vec![lstm.zero_state(1)?];
        for &inp in inp_sequence.iter() {
            println!("Input: {:?}", inp);
            let inp = Tensor::new(&[[inp, inp * 0.5]], &device)?;
            // Print the input tensor
            // Convert the tensor to a Vec and print its values
            match inp.to_vec2::<f32>() {
                Ok(values) => println!("Input tensor values: {:?}", values),
                Err(e) => println!("Error converting tensor to vec: {:?}", e),
            }
            let state = lstm.step(&inp, &states.last().unwrap())?;
            states.push(state)
        }

        // Apply the backward lstm and collect the results in states_rv
        print!("Applying backward LSTM...");
        let mut states_rev = vec![lstm.zero_state(1)?];
        for &inp in inp_sequence.iter().rev() {
            println!("Input: {:?}", inp);
            let inp = Tensor::new(&[[inp, inp * 0.5]], &device)?;
            // Convert the tensor to a Vec and print its values
            match inp.to_vec2::<f32>() {
                Ok(values) => println!("Input tensor values: {:?}", values),
                Err(e) => println!("Error converting tensor to vec: {:?}", e),
            }
            let state = lstm_rev.step(&inp, &states_rev.last().unwrap())?;
            states_rev.push(state)
        }

        // Merge the results together
        let states = states
            .into_iter()
            .zip(states_rev.into_iter().rev())
            .collect::<Vec<_>>();

        println!("States: {:?}", states);
        Ok(())
    }
}
