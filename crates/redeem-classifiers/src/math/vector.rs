//! 1D array utilities (Array1) and small vector helpers.
//!
//! Provides `Array1<T>` with convenience constructors, element-wise
//! operations and a few numeric helpers (mean, dot). The implementation
//! is deliberately small and suitable for unit tests and examples.
use std::fmt;
use std::iter::FromIterator;
use std::ops::{BitAnd, BitOr, Index, IndexMut};
use std::slice::{Iter, IterMut};

use num_traits::{One, Zero};

#[derive(Clone, Debug, PartialEq)]
pub struct Array1<T> {
    data: Vec<T>,
}

impl<T> Array1<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        Self::new(data)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn shape(&self) -> (usize,) {
        (self.len(),)
    }

    pub fn mapv<U, F>(&self, f: F) -> Array1<U>
    where
        F: FnMut(&T) -> U,
    {
        Array1::from_vec(self.data.iter().map(f).collect())
    }

    pub fn select(&self, indices: &[usize]) -> Array1<T>
    where
        T: Clone,
    {
        let mut selected = Vec::with_capacity(indices.len());
        for &idx in indices {
            selected.push(self.data[idx].clone());
        }
        Array1::from_vec(selected)
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }
}

impl<T> Array1<T>
where
    T: Clone,
{
    pub fn from_elem(len: usize, value: T) -> Self {
        Array1::from_vec(vec![value; len])
    }
}

impl<T> Array1<T>
where
    T: Clone + Zero,
{
    pub fn zeros(len: usize) -> Self {
        Array1::from_vec(vec![T::zero(); len])
    }
}

impl<T> Array1<T>
where
    T: Clone + One,
{
    pub fn ones(len: usize) -> Self {
        Array1::from_vec(vec![T::one(); len])
    }
}

impl<T> From<Vec<T>> for Array1<T> {
    fn from(value: Vec<T>) -> Self {
        Array1::from_vec(value)
    }
}

impl<T> From<Array1<T>> for Vec<T> {
    fn from(value: Array1<T>) -> Self {
        value.data
    }
}

impl<T> FromIterator<T> for Array1<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Array1::from_vec(iter.into_iter().collect())
    }
}

impl<T> Index<usize> for Array1<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Array1<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'b> BitAnd<&'b Array1<bool>> for &Array1<bool> {
    type Output = Array1<bool>;

    fn bitand(self, rhs: &'b Array1<bool>) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Bitwise and requires arrays of equal length"
        );
        Array1::from_vec(self.iter().zip(rhs.iter()).map(|(a, b)| *a && *b).collect())
    }
}

impl<'b> BitOr<&'b Array1<bool>> for &Array1<bool> {
    type Output = Array1<bool>;

    fn bitor(self, rhs: &'b Array1<bool>) -> Self::Output {
        assert_eq!(
            self.len(),
            rhs.len(),
            "Bitwise or requires arrays of equal length"
        );
        Array1::from_vec(self.iter().zip(rhs.iter()).map(|(a, b)| *a || *b).collect())
    }
}

impl Array1<f64> {
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            None
        } else {
            Some(self.iter().copied().sum::<f64>() / self.len() as f64)
        }
    }

    pub fn dot(&self, other: &Array1<f64>) -> f64 {
        assert_eq!(
            self.len(),
            other.len(),
            "Dot product requires equal length vectors"
        );
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            unsafe { dot_simd_f64(self.as_slice(), other.as_slice()) }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            dot_scalar_f64(self.as_slice(), other.as_slice())
        }
    }
}

fn dot_scalar_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
unsafe fn dot_simd_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let mut acc = _mm_setzero_pd();

    while i + 2 <= lhs.len() {
        let a = _mm_loadu_pd(lhs.as_ptr().add(i));
        let b = _mm_loadu_pd(rhs.as_ptr().add(i));
        acc = _mm_add_pd(acc, _mm_mul_pd(a, b));
        i += 2;
    }

    let mut buffer = [0f64; 2];
    _mm_storeu_pd(buffer.as_mut_ptr(), acc);
    let mut sum = buffer.iter().sum::<f64>();

    while i < lhs.len() {
        sum += lhs[i] * rhs[i];
        i += 1;
    }

    sum
}

impl<T: fmt::Display> fmt::Display for Array1<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (idx, value) in self.data.iter().enumerate() {
            write!(f, "{}", value)?;
            if idx + 1 != self.data.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}
