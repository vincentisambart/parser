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
        loop {
            match self.peek() {
                Some(x) => if matcher(x) {
                    self.next()
                } else {
                    break;
                },
                None => break,
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
        loop {
            match self.peek() {
                Some(c) => if matcher(c) {
                    string.push(*c);
                } else {
                    break;
                },
                None => break,
            };
            self.next();
        }
    }
}
