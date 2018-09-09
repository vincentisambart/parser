use std::iter::Peekable;

pub trait Scan {
    type Item;

    fn scan_one<F>(&mut self, matcher: F) -> Option<Self::Item>
    where
        F: FnMut(&Self::Item) -> bool;

    fn skip_matching<F>(&mut self, matcher: F)
    where
        F: FnMut(&Self::Item) -> bool;

    fn check_one<F>(&mut self, matcher: F) -> Option<&Self::Item>
    where
        F: FnMut(&Self::Item) -> bool;
}

impl<I> Scan for Peekable<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn scan_one<F>(&mut self, mut matcher: F) -> Option<Self::Item>
    where
        F: FnMut(&Self::Item) -> bool,
    {
        match self.peek() {
            Some(x) => if matcher(x) {
                self.next()
            } else {
                None
            },
            None => None,
        }
    }

    fn skip_matching<F>(&mut self, mut matcher: F)
    where
        F: FnMut(&Self::Item) -> bool,
    {
        while let Some(x) = self.peek() {
            if matcher(x) {
                self.next()
            } else {
                break;
            };
        }
    }

    fn check_one<F>(&mut self, mut matcher: F) -> Option<&Self::Item>
    where
        F: FnMut(&Self::Item) -> bool,
    {
        match self.peek() {
            Some(x) => if matcher(x) {
                Some(x)
            } else {
                None
            },
            None => None,
        }
    }
}

pub trait ScanErr<T, E> {
    fn scan_one_err<F>(&mut self, matcher: F) -> Result<Option<T>, E>
    where
        F: FnMut(&T) -> bool;
}

impl<I, T, E> ScanErr<T, E> for Peekable<I>
where
    I: Iterator<Item = Result<T, E>>,
{
    fn scan_one_err<F>(&mut self, mut matcher: F) -> Result<Option<T>, E>
    where
        F: FnMut(&T) -> bool,
    {
        match self.peek() {
            Some(Ok(x)) => if matcher(x) {
                if let Some(Ok(x)) = self.next() {
                    Ok(Some(x))
                } else {
                    unreachable!()
                }
            } else {
                Ok(None)
            },
            Some(Err(_)) => {
                if let Some(Err(err)) = self.next() {
                    Err(err)
                } else {
                    unreachable!()
                }
            }
            None => Ok(None),
        }
    }
}

pub trait ScanPush {
    fn scan_push<F>(&mut self, matcher: F, string: &mut String)
    where
        F: FnMut(&char) -> bool;
}

impl<I> ScanPush for Peekable<I>
where
    I: Iterator<Item = char>,
{
    fn scan_push<F>(&mut self, mut matcher: F, string: &mut String)
    where
        F: FnMut(&char) -> bool,
    {
        while let Some(c) = self.peek() {
            if matcher(c) {
                string.push(*c);
            } else {
                break;
            };
            self.next();
        }
    }
}
