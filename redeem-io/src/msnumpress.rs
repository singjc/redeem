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

    #[cfg(feature = "parquet")]
    #[test]
    fn test_decode_linear_zlib_example_rt() -> Result<()> {
        use flate2::read::ZlibDecoder;
        use std::io::Read;

        let rt_zlib: &[u8] = b"x\x9csPa\x00\x83\xec\x1c\x06\x86^ n\x97\xff\xd1\xd1Q\xf8\xbf\xa3\x03\xce@\xe5\x81\x19\x98\x8ap\x08cQ\x84*\x0c\x00\xec5A\xf5";
        let mut decoder = ZlibDecoder::new(rt_zlib);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;

        let decoded = decode_linear(&buf)?;
        let expected: &[f64] = &[
            2775.5, 2778.9, 2782.3, 2785.8, 2789.2, 2792.6, 2796.0, 2799.4,
            2802.8, 2806.2, 2809.7, 2813.1, 2816.5, 2819.9, 2823.3, 2826.7,
            2830.1, 2833.6, 2837.0, 2840.4, 2843.8, 2847.2, 2850.6, 2854.0,
            2857.5, 2860.9, 2864.3, 2867.7, 2871.1, 2874.5, 2877.9, 2881.3,
            2884.8, 2888.2, 2891.6, 2895.0, 2898.4, 2901.8, 2905.2, 2908.7,
            2912.1, 2915.5, 2918.9, 2922.3, 2925.7, 2929.1, 2932.6, 2936.0,
            2939.4, 2942.8, 2946.2, 2949.6, 2953.0, 2956.5, 2959.9, 2963.3,
            2966.7, 2970.1, 2973.5, 2976.9, 2980.3, 2983.8, 2987.2, 2990.6,
            2994.0, 2997.4, 3000.8, 3004.2, 3007.7, 3011.1, 3014.5, 3017.9,
            3021.3, 3024.7, 3028.1, 3031.6, 3035.0, 3038.4, 3041.8, 3045.2,
            3048.6, 3052.0, 3055.5, 3058.9, 3062.3, 3065.7, 3069.1, 3072.5,
            3075.9, 3079.4, 3082.8, 3086.2, 3089.6, 3093.0, 3096.4, 3099.8,
            3103.2, 3106.7, 3110.1, 3113.5, 3116.9, 3120.3, 3123.7, 3127.1,
            3130.6, 3134.0, 3137.4, 3140.8, 3144.2, 3147.6, 3151.0, 3154.5,
            3157.9, 3161.3, 3164.7, 3168.1, 3171.5, 3174.9, 3178.4, 3181.8,
            3185.2, 3188.6, 3192.0, 3195.4, 3198.8, 3202.2, 3205.7, 3209.1,
            3212.5, 3215.9, 3219.3, 3222.7, 3226.1, 3229.6, 3233.0, 3236.4,
            3239.8, 3243.2, 3246.6, 3250.0, 3253.5, 3256.9, 3260.3, 3263.7,
            3267.1, 3270.5, 3273.9, 3277.4, 3280.8, 3284.2, 3287.6, 3291.0,
            3294.4, 3297.8, 3301.3, 3304.7, 3308.1, 3311.5, 3314.9, 3318.3,
            3321.7, 3325.1, 3328.6, 3332.0, 3335.4, 3338.8, 3342.2, 3345.6,
            3349.0, 3352.5, 3355.9, 3359.3, 3362.7, 3366.1, 3369.5, 3372.9,
        ];
        assert_eq!(decoded.len(), expected.len());
        for (a, b) in decoded.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_decode_slof_zlib_example_intensity() -> Result<()> {
        use flate2::read::ZlibDecoder;
        use std::io::Read;

        let int_zlib: &[u8] = b"x\x9csx\xa0s\x80\x01\r\x04\xfcA\x17!\x05\xfc\xfb\x8f]\xbc\xa3\x15S\xac+\x8bXS\xcf\x1c\xc1/\x1f\x93\x0ec\xbd\xde\x82[\x95x\x1c\x88\xdco\r\xe3+\x1f&\xce\xf64WT>\xba\x1f\r\xa3\x08\x99\x00\x00\xe6\xa3\x10\xd1";
        let mut decoder = ZlibDecoder::new(int_zlib);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;

        let decoded = decode_slof(&buf)?;
        let expected: &[f64] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.02790256, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.23062287, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.80650141,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.27809868,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            3.57615785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2228015, 0.0, 0.0, 0.0,
            0.0, 3.0476905, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.06913939, 0.0, 0.0, 0.58680397, 0.0,
            0.0, 0.0, 0.0, 3.51782167, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.70969653, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 6.23062287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00773129,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(decoded.len(), expected.len());
        for (a, b) in decoded.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
        Ok(())
    }
}
