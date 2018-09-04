// Notes
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
mod lex;
use crate::lex::LexError;
use crate::lex::TokenIter;

fn main() -> Result<(), LexError> {
    let iter = TokenIter::from(
        r#"# 1 "test.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 341 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "test.c" 2
# 1 "./test.h" 1
int foo(void);
# 2 "test.c" 2
int foo(void) {
  return 10;
}
"#,
    );
    for token in iter {
        println!("token: {:?}", token?);
    }
    Ok(())
}
