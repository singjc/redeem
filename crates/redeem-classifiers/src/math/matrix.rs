//! 2D array utilities (Array2) and shape error handling.
//!
//! This module implements a compact `Array2<T>` with basic indexing,
//! slicing and selection helpers used by the rest of the crate. It's
//! intentionally minimal compared to full ndarray-like crates.
use std::error::Error;
use std::fmt;
use std::ops::{Index, IndexMut, RangeBounds};

use crate::math::vector::Array1;

#[derive(Clone, Debug, PartialEq)]
pub struct Array2<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> Array2<T> {
    pub fn from_shape_vec(shape: (usize, usize), data: Vec<T>) -> Result<Self, ShapeError> {
        let (rows, cols) = shape;
        if data.len() != rows * cols {
            return Err(ShapeError {
                rows,
                cols,
                len: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, ShapeError> {
        Self::from_shape_vec((rows, cols), data)
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }

    pub fn ncols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    fn offset(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub fn row_slice(&self, row: usize) -> &[T] {
        let start = self.offset(row, 0);
        &self.data[start..start + self.cols]
    }

    pub fn column(&self, col: usize) -> Array1<T>
    where
        T: Clone,
    {
        assert!(col < self.cols, "column index out of bounds");
        let mut values = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            values.push(self[(row, col)].clone());
        }
        Array1::from_vec(values)
    }

    pub fn select_rows(&self, indices: &[usize]) -> Array2<T>
    where
        T: Clone,
    {
        let mut data = Vec::with_capacity(indices.len() * self.cols);
        for &row in indices {
            let slice = self.row_slice(row);
            data.extend_from_slice(slice);
        }
        Array2 {
            data,
            rows: indices.len(),
            cols: self.cols,
        }
    }

    pub fn select_columns<R>(&self, range: R) -> Array2<T>
    where
        R: RangeBounds<usize>,
        T: Clone,
    {
        use std::ops::Bound;

        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
        };

        let end = match range.end_bound() {
            Bound::Unbounded => self.cols,
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
        };

        assert!(
            start <= end && end <= self.cols,
            "column slice out of bounds"
        );

        let new_cols = end - start;
        let mut data = Vec::with_capacity(self.rows * new_cols);
        for row in 0..self.rows {
            let slice = &self.row_slice(row)[start..end];
            data.extend_from_slice(slice);
        }

        Array2 {
            data,
            rows: self.rows,
            cols: new_cols,
        }
    }

    pub fn mapv<U, F>(&self, f: F) -> Array2<U>
    where
        F: FnMut(&T) -> U,
    {
        Array2 {
            data: self.data.iter().map(f).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }
}

impl<T> Index<(usize, usize)> for Array2<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let offset = self.offset(index.0, index.1);
        &self.data[offset]
    }
}

impl<T> IndexMut<(usize, usize)> for Array2<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let offset = self.offset(index.0, index.1);
        &mut self.data[offset]
    }
}

#[derive(Debug, Clone)]
pub struct ShapeError {
    rows: usize,
    cols: usize,
    len: usize,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid shape ({}, {}) for buffer of length {}",
            self.rows, self.cols, self.len
        )
    }
}

impl Error for ShapeError {}
