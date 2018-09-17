/// An iterator-like trait for which next can fail.
pub trait FailableIterator {
    type Item;
    type Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error>;

    fn peekable(self) -> FailablePeekable<Self>
    where
        Self: Sized,
    {
        FailablePeekable {
            iter: self,
            peeked: None,
        }
    }

    fn collect(mut self) -> Result<Vec<Self::Item>, Self::Error>
    where
        Self: Sized,
    {
        let mut vec = Vec::new();
        while let Some(item) = self.next()? {
            vec.push(item);
        }
        Ok(vec)
    }
}

/// Peekable for FailableIterator. Note that peek can also fail.
pub struct FailablePeekable<I: FailableIterator> {
    iter: I,
    peeked: Option<Option<I::Item>>,
}

impl<I: FailableIterator> FailablePeekable<I> {
    pub fn peek(&mut self) -> Result<Option<&I::Item>, I::Error> {
        if self.peeked.is_none() {
            self.peeked = Some(self.iter.next()?);
        }
        match self.peeked {
            Some(Some(ref value)) => Ok(Some(value)),
            Some(None) => Ok(None),
            _ => unreachable!(),
        }
    }

    pub fn next_if<F>(&mut self, mut predicate: F) -> Result<Option<I::Item>, I::Error>
    where
        F: FnMut(&I::Item) -> bool,
    {
        match self.peek()? {
            Some(x) => if predicate(x) {
                self.next()
            } else {
                Ok(None)
            },
            None => Ok(None),
        }
    }

    pub fn advance_if<F>(&mut self, predicate: F) -> Result<bool, I::Error>
    where
        F: FnMut(&I::Item) -> bool,
    {
        Ok(self.next_if(predicate)?.is_some())
    }
}

impl<I: FailableIterator> FailableIterator for FailablePeekable<I> {
    type Item = I::Item;
    type Error = I::Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        match self.peeked.take() {
            Some(peeked) => Ok(peeked),
            None => self.iter.next(),
        }
    }
}

/// An iterator-like trait with the following properties:
/// - can fail (next can return an error)
/// - next can return a variable number of items (including 0, and that doesn't mean the end of the stream).
pub trait VarRateFailableIterator {
    type Item;
    type Error;
    type VarRateItemsIter: Iterator<Item = Self::Item>;

    fn next(&mut self) -> Result<Option<Self::VarRateItemsIter>, Self::Error>;

    fn collect(mut self) -> Result<Vec<Self::Item>, Self::Error>
    where
        Self: Sized,
    {
        let mut vec = Vec::new();
        while let Some(iter) = self.next()? {
            let (to_reserve, _) = iter.size_hint();
            vec.reserve(to_reserve);
            for item in iter {
                vec.push(item);
            }
        }
        Ok(vec)
    }
}

/// An adapter from VarRateFailableIterator to a normal (fixed rate) FailableIterator.
pub struct FixedRateFailableIterator<I: VarRateFailableIterator> {
    iter: I,
    previous: Option<I::VarRateItemsIter>,
}

impl<I: VarRateFailableIterator> FailableIterator for FixedRateFailableIterator<I> {
    type Item = I::Item;
    type Error = I::Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if let Some(ref mut previous) = self.previous {
            if let Some(value) = previous.next() {
                return Ok(Some(value));
            }
            self.previous = None;
        }
        loop {
            self.previous = self.iter.next()?;
            match self.previous {
                None => return Ok(None),
                Some(ref mut previous) => {
                    if let Some(value) = previous.next() {
                        return Ok(Some(value));
                    }
                }
            }
        }
    }
}
