// TODO:
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
// - Add position to declarations
// - Variable initialization
// - Error on _Thread_local in typedefs and other forbidden places

mod error;
mod failable;
mod lex;
mod parser;
mod peeking;

use crate::error::ParseError;
use crate::failable::FailableIterator;
use crate::parser::Parser;

fn main() -> Result<(), ParseError> {
    let mut parser = Parser::from_code(r#"x;"#);
    // let mut parser = Parser::from_code(include_str!("../y.pp.m"));
    while let Some(decl) = parser.next()? {
        println!("Declaration: {:?}", decl);
    }
    Ok(())
}
