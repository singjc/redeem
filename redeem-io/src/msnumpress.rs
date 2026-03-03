// redeem-io/src/msnumpress.rs

use anyhow::{bail, Result};

fn decode_fixed_point(data: &[u8]) -> Result<f64> {
    if data.len() < 8 {
        bail!("msnumpress: not enough bytes to decode fixed point");
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[..8]);
    Ok(f64::from_be_bytes(buf))
}

pub fn decode_int(
    data: &[u8],
    di: &mut usize,
    max_di: usize,
    half: &mut usize,
    res: &mut u32,
) -> Result<()> {
    let n: usize;
    let mask: u32;
    let mut m: u32;
    let head: u8;
    let mut hb: u8;

    if *half == 0 {
        head = data[*di] >> 4;
    } else {
        head = data[*di] & 0xf;
        *di += 1;
    }
    *half = 1 - *half;
    *res = 0;

    if head <= 8 {
        n = head as usize;
    } else {
        n = (head - 8) as usize;
        mask = 0xf0000000;
        for i in 0..n {
            m = mask >> (4 * i);
            *res |= m;
        }
    }

    if n == 8 {
        return Ok(());
    }

    if *di + ((8 - n) - (1 - *half)) / 2 >= max_di {
        bail!("msnumpress decode_int: corrupt input");
    }

    for i in n..8 {
        if *half == 0 {
            hb = data[*di] >> 4;
        } else {
            hb = data[*di] & 0xf;
            *di += 1;
        }
        *res |= (hb as u32) << ((i - n) * 4);
        *half = 1 - *half;
    }

    Ok(())
}

pub fn decode_linear(data: &[u8]) -> Result<Vec<f64>> {
    let data_size = data.len();
    if data_size < 8 {
        bail!("msnumpress decode_linear: data too small");
    }

    let fixed_point = decode_fixed_point(data)?;

    if data_size < 12 {
        bail!("msnumpress decode_linear: not enough bytes for first value");
    }

    let mut result = vec![0.0; (data_size - 8) * 2];

    let mut ints = [0i64; 3];
    let mut init: u8;
    ints[1] = 0;
    for i in 0..4 {
        init = data[8 + i];
        ints[1] |= ((init & 0xff) as i64) << (i * 8);
    }
    result[0] = ints[1] as f64 / fixed_point;

    if data_size == 12 {
        result.truncate(1);
        return Ok(result);
    }
    if data_size < 16 {
        bail!("msnumpress decode_linear: not enough bytes for second value");
    }

    ints[2] = 0;
    for i in 0..4 {
        init = data[12 + i];
        ints[2] |= ((init & 0xff) as i64) << (i * 8);
    }
    result[1] = ints[2] as f64 / fixed_point;

    let mut ri = 2usize;
    let mut half = 0usize;
    let mut di = 16usize;
    let mut buff: u32 = 0;

    while di < data_size {
        if di == (data_size - 1) && half == 1 {
            if (data[di] & 0xf) == 0x0 {
                break;
            }
        }

        ints[0] = ints[1];
        ints[1] = ints[2];

        decode_int(data, &mut di, data_size, &mut half, &mut buff)?;
        let diff = buff as i32;
        let extrapol = ints[1] + (ints[1] - ints[0]);
        let y = extrapol + diff as i64;

        result[ri] = y as f64 / fixed_point;
        ri += 1;
        ints[2] = y;
    }

    result.truncate(ri);
    Ok(result)
}

pub fn decode_slof(data: &[u8]) -> Result<Vec<f64>> {
    let data_size = data.len();
    if data_size < 8 {
        bail!("msnumpress decode_slof: data too small");
    }
    let fixed_point = decode_fixed_point(data)?;
    let payload = data_size - 8;
    let trimmed = payload - (payload % 2);
    let end = 8 + trimmed;
    let mut result = vec![0.0; trimmed / 2];
    let mut ri = 0usize;
    for i in (8..end).step_by(2) {
        let x = (data[i] as u16) | ((data[i + 1] as u16) << 8);
        result[ri] = (x as f64 / fixed_point).exp() - 1.0;
        ri += 1;
    }
    result.truncate(ri);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATA: [f64; 4] = [100.0, 101.0, 102.0, 103.0];
    const DATA_SLOF: [f64; 4] = [100.0, 200.0, 300.00005, 400.00010];

    const LINEAR_RESULT: [u8; 17] = [
        0x40, 0xf8, 0x6a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x96, 0x98, 0x00, 0x20, 0x1d,
        0x9a, 0x00, 0x88,
    ];

    const SLOF_RESULT: &[u8] = &[
        0x40, 0xc3, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0xb4, 0x29, 0xcf, 0xef, 0xde,
        0x24, 0xea,
    ];

    #[test]
    fn test_decode_linear() {
        let decoded = decode_linear(&LINEAR_RESULT).unwrap();
        assert_eq!(decoded.len(), DATA.len());
        for (a, b) in decoded.iter().zip(DATA.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_decode_slof() {
        let decoded = decode_slof(SLOF_RESULT).unwrap();
        assert_eq!(decoded.len(), DATA_SLOF.len());
        for (a, b) in decoded.iter().zip(DATA_SLOF.iter()) {
            assert!((a - b).abs() < 1.0);
        }
    }
}
