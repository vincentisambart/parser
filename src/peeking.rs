use std::iter::Peekable;

pub trait Peeking<I: Iterator> {
    fn next_if<F>(&mut self, predicate: F) -> Option<I::Item>
    where
        F: FnMut(&I::Item) -> bool;

    fn advance_if<F>(&mut self, predicate: F) -> bool
    where
        F: FnMut(&I::Item) -> bool;

    fn advance_while<F>(&mut self, predicate: F)
    where
        F: FnMut(&I::Item) -> bool;

    fn peeking_does_match<F>(&mut self, predicate: F) -> bool
    where
        F: FnMut(&I::Item) -> bool;

    fn push_while<F>(&mut self, string: &mut String, predicate: F)
    where
        F: FnMut(&char) -> bool,
        I: Iterator<Item = char>;
}

impl<I> Peeking<I> for Peekable<I>
where
    I: Iterator,
{
    fn next_if<F>(&mut self, mut predicate: F) -> Option<I::Item>
    where
        F: FnMut(&I::Item) -> bool,
    {
        match self.peek() {
            Some(x) => if predicate(x) {
                self.next()
            } else {
                None
            },
            None => None,
        }
    }

    fn advance_if<F>(&mut self, predicate: F) -> bool
    where
        F: FnMut(&I::Item) -> bool,
    {
        self.next_if(predicate).is_some()
    }

    fn advance_while<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&I::Item) -> bool,
    {
        while let Some(x) = self.peek() {
            if predicate(x) {
                self.next()
            } else {
                break;
            };
        }
    }

    fn peeking_does_match<F>(&mut self, mut predicate: F) -> bool
    where
        F: FnMut(&I::Item) -> bool,
    {
        match self.peek() {
            Some(x) => predicate(x),
            None => false,
        }
    }

    fn push_while<F>(&mut self, string: &mut String, mut predicate: F)
    where
        F: FnMut(&char) -> bool,
        I: Iterator<Item = char>,
    {
        while let Some(c) = self.peek() {
            if predicate(c) {
                string.push(*c);
            } else {
                break;
            };
            self.next();
        }
    }
}
