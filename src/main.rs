// Notes
// - For testing pragmas, have a look at clang's test/Sema/pragma-align-packed.c
mod lex;
mod scan;
use bitflags::bitflags;
use crate::lex::{Keyword, LexError, Position, PositionedToken, Punctuator, Token, TokenIter};
use crate::scan::Peeking;
use std::collections::HashMap;
use std::iter::Peekable;
use std::rc::Rc;

trait PeekingToken {
    fn advance_if_kw(&mut self, kw: Keyword) -> bool;
    fn advance_if_punc(&mut self, punc: Punctuator) -> bool;
    fn next_if_any_ident(&mut self) -> Option<String>;
}

impl<I> PeekingToken for Peekable<I>
where
    I: Iterator<Item = PositionedToken>,
{
    fn advance_if_kw(&mut self, kw: Keyword) -> bool {
        self.advance_if(|token| match token {
            PositionedToken(Token::Keyword(x), _) if *x == kw => true,
            _ => false,
        })
    }

    fn advance_if_punc(&mut self, punc: Punctuator) -> bool {
        self.advance_if(|token| match token {
            PositionedToken(Token::Punctuator(x), _) if *x == punc => true,
            _ => false,
        })
    }

    fn next_if_any_ident(&mut self) -> Option<String> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Identifier(_), _) => true,
            _ => false,
        });
        match token {
            Some(PositionedToken(Token::Identifier(ident), _)) => Some(ident),
            Some(_) => unreachable!(),
            None => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PrimitiveType {
    Void,
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    Float,
    Double,
    LongDouble,
    Bool,
    FloatComplex,
    DoubleComplex,
    LongDoubleComplex,
}

bitflags! {
    struct TypeQualifiers: u8 {
        const CONST    = 1;
        const VOLATILE = 1 << 1;
        const RESTRICT = 1 << 2;
        const ATOMIC   = 1 << 3;
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Primitive(PrimitiveType),
    Pointer(Box<QualType>),
    FuncPointer(Box<FuncType>),
    UserDefined(String, Rc<QualType>),
}

#[derive(Debug, Clone, PartialEq)]
struct QualType {
    ty: Type,
    qualifiers: TypeQualifiers,
}

impl QualType {
    fn new(ty: Type, qualifiers: TypeQualifiers) -> QualType {
        QualType { ty, qualifiers }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum FuncArgs {
    Undefined,
    Defined {
        args: Vec<(String, Type)>,
        var_args: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct FuncType {
    ret_type: QualType,
    args: FuncArgs,
}

impl FuncType {
    fn new(ret_type: QualType, args: FuncArgs) -> FuncType {
        FuncType { ret_type, args }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ExternalDecl {
    VarDecl(String, QualType),
    FuncDef(String, FuncType),
    Nothing,
}

#[derive(Debug, Clone)]
pub enum ParseError {
    LexError(LexError),
    ExpectingToken(Token), // TODO: Should have position
    UnexpectedToken(Token, Position),
}

impl From<LexError> for ParseError {
    fn from(error: LexError) -> Self {
        ParseError::LexError(error)
    }
}

struct Parser<I: Iterator<Item = PositionedToken>> {
    iter: Peekable<I>,
    types_stack: Vec<HashMap<String, Rc<Type>>>,
}

impl Parser<std::vec::IntoIter<PositionedToken>> {
    fn from_code(code: &str) -> Result<Self, ParseError> {
        let iter = TokenIter::from(code);
        let mut tokens = Vec::new();
        for result in iter {
            match result {
                Ok(token) => tokens.push(token),
                Err(err) => return Err(err.into()),
            }
        }
        Ok(Self::from(tokens.into_iter()))
    }
}

impl<I> Parser<I>
where
    I: Iterator<Item = PositionedToken>,
{
    fn from(iter: I) -> Parser<I> {
        Parser {
            iter: iter.peekable(),
            types_stack: Vec::new(),
        }
    }

    fn type_by_name(&self, name: &str) -> Option<Rc<Type>> {
        for types in self.types_stack.iter().rev() {
            if let Some(ty) = types.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    fn is_type_name(&self, name: &str) -> bool {
        self.type_by_name(name).is_some()
    }

    fn read_type_qualifier(&mut self) -> Option<TypeQualifiers> {
        if self.iter.advance_if_kw(Keyword::Const) {
            Some(TypeQualifiers::CONST)
        } else if self.iter.advance_if_kw(Keyword::Volatile) {
            Some(TypeQualifiers::VOLATILE)
        } else if self.iter.advance_if_kw(Keyword::Restrict) {
            Some(TypeQualifiers::RESTRICT)
        } else if self.iter.advance_if_kw(Keyword::Atomic) {
            Some(TypeQualifiers::ATOMIC)
        } else {
            None
        }
    }

    fn read_base_type(&mut self) -> Option<Type> {
        if let Some(prim) = self.read_primitive_type() {
            return Some(Type::Primitive(prim));
        };
        match self.iter.peek() {
            Some(PositionedToken(Token::Identifier(identifier), _)) => {
                // TODO: Clone should not be needed (moving is_type_name to another struct might do the trick)
                let copy = identifier.clone();
                if self.is_type_name(copy.as_ref()) {
                    panic!("TODO")
                } else {
                    None
                }
            }
            Some(PositionedToken(Token::Keyword(Keyword::Int), _)) => {
                self.iter.next();
                Some(Type::Primitive(PrimitiveType::Int))
            }
            _ => None,
        }
    }

    fn read_primitive_type(&mut self) -> Option<PrimitiveType> {
        if self.iter.advance_if_kw(Keyword::Void) {
            Some(PrimitiveType::Void)
        } else if self.iter.advance_if_kw(Keyword::Char) {
            Some(PrimitiveType::Char)
        } else if self.iter.advance_if_kw(Keyword::Signed) {
            if self.iter.advance_if_kw(Keyword::Char) {
                Some(PrimitiveType::SignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short) {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::Short)
            } else if self.iter.advance_if_kw(Keyword::Long) {
                if self.iter.advance_if_kw(Keyword::Long) {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::LongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::Long)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::Int)
            }
        } else if self.iter.advance_if_kw(Keyword::Unsigned) {
            if self.iter.advance_if_kw(Keyword::Char) {
                Some(PrimitiveType::UnsignedChar)
            } else if self.iter.advance_if_kw(Keyword::Short) {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::UnsignedShort)
            } else if self.iter.advance_if_kw(Keyword::Long) {
                if self.iter.advance_if_kw(Keyword::Long) {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::UnsignedLongLong)
                } else {
                    self.iter.advance_if_kw(Keyword::Int);
                    Some(PrimitiveType::UnsignedLong)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::UnsignedInt)
            }
        } else if self.iter.advance_if_kw(Keyword::Short) {
            self.iter.advance_if_kw(Keyword::Int);
            Some(PrimitiveType::Short)
        } else if self.iter.advance_if_kw(Keyword::Int) {
            Some(PrimitiveType::Int)
        } else if self.iter.advance_if_kw(Keyword::Long) {
            if self.iter.advance_if_kw(Keyword::Long) {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::LongLong)
            } else if self.iter.advance_if_kw(Keyword::Double) {
                if self.iter.advance_if_kw(Keyword::Complex) {
                    Some(PrimitiveType::LongDoubleComplex)
                } else {
                    Some(PrimitiveType::LongDouble)
                }
            } else {
                self.iter.advance_if_kw(Keyword::Int);
                Some(PrimitiveType::Long)
            }
        } else if self.iter.advance_if_kw(Keyword::Float) {
            if self.iter.advance_if_kw(Keyword::Complex) {
                Some(PrimitiveType::FloatComplex)
            } else {
                Some(PrimitiveType::Float)
            }
        } else if self.iter.advance_if_kw(Keyword::Double) {
            if self.iter.advance_if_kw(Keyword::Complex) {
                Some(PrimitiveType::DoubleComplex)
            } else {
                Some(PrimitiveType::Double)
            }
        } else if self.iter.advance_if_kw(Keyword::Bool) {
            Some(PrimitiveType::Bool)
        } else {
            None
        }
    }

    fn expect_token(&mut self, expected_token: Token) -> Result<(), ParseError> {
        match self.iter.next() {
            Some(PositionedToken(token, position)) => if token == expected_token {
                Ok(())
            } else {
                Err(ParseError::UnexpectedToken(token, position))
            },
            None => Err(ParseError::ExpectingToken(expected_token)),
        }
    }

    // Should be called just after having read an opening parenthesis.
    fn read_func_args(&mut self) -> Result<FuncArgs, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightParenthesis) {
            return Ok(FuncArgs::Undefined);
        }
        // TODO: Handle parameters
        panic!("TODO: Handle parameters")
    }

    fn read_next_external_declaration(&mut self) -> Result<Option<ExternalDecl>, ParseError> {
        if self.iter.peek().is_none() {
            return Ok(None);
        }
        let base_qual_ty = {
            let mut qualifiers = TypeQualifiers::empty();
            while let Some(qualifier) = self.read_type_qualifier() {
                qualifiers |= qualifier;
            }
            let ty = self
                .read_base_type()
                .unwrap_or(Type::Primitive(PrimitiveType::Int));
            while let Some(qualifier) = self.read_type_qualifier() {
                qualifiers |= qualifier;
            }
            QualType::new(ty, qualifiers)
        };
        let mut ptr_qualifs = Vec::new();
        let mut ptr_qualifs_stack = Vec::new();
        let ident = loop {
            if self.iter.advance_if_punc(Punctuator::Star) {
                let mut qualifiers = TypeQualifiers::empty();
                while let Some(qualifier) = self.read_type_qualifier() {
                    qualifiers |= qualifier;
                }
                ptr_qualifs.push(qualifiers);
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis) {
                ptr_qualifs_stack.push(ptr_qualifs);
                ptr_qualifs = Vec::new();
            } else if let Some(ident) = self.iter.next_if_any_ident() {
                ptr_qualifs_stack.push(ptr_qualifs);
                break ident;
            } else if self.iter.advance_if_punc(Punctuator::Semicolon) {
                if !ptr_qualifs_stack.is_empty() {
                    panic!("TODO");
                }
                return Ok(Some(ExternalDecl::Nothing));
            } else {
                self.expect_token(Token::Punctuator(Punctuator::Semicolon))?;
            }
        };
        let mut ptr_qualifs_reversed_stack = Vec::new();
        loop {
            let ptr_qualifs = if let Some(ptr_qualifs) = ptr_qualifs_stack.pop() {
                ptr_qualifs
            } else {
                panic!("TODO: Incorrect decl");
            };
            if self.iter.advance_if_punc(Punctuator::LeftParenthesis) {
                let func_args = self.read_func_args()?;
                ptr_qualifs_reversed_stack.push((ptr_qualifs, Some(func_args)));
            } else {
                ptr_qualifs_reversed_stack.push((ptr_qualifs, None));
            }
            if ptr_qualifs_stack.is_empty() {
                break;
            } else {
                self.expect_token(Token::Punctuator(Punctuator::RightParenthesis))?;
            }
        }
        assert!(ptr_qualifs_stack.is_empty());
        drop(ptr_qualifs_stack);

        self.expect_token(Token::Punctuator(Punctuator::Semicolon))?;

        enum FuncOrQualType {
            Func(FuncType),
            QualType(QualType),
        }

        let mut func_or_qual_ty = FuncOrQualType::QualType(base_qual_ty);
        while let Some((ptr_qualifs, func_args)) = ptr_qualifs_reversed_stack.pop() {
            for qualifiers in ptr_qualifs {
                func_or_qual_ty = match func_or_qual_ty {
                    FuncOrQualType::QualType(qual_ty) => FuncOrQualType::QualType(QualType::new(
                        Type::Pointer(Box::new(qual_ty)),
                        qualifiers,
                    )),
                    FuncOrQualType::Func(func) => FuncOrQualType::QualType(QualType::new(
                        Type::FuncPointer(Box::new(func)),
                        qualifiers,
                    )),
                };
            }
            if let Some(func_args) = func_args {
                func_or_qual_ty = match func_or_qual_ty {
                    FuncOrQualType::QualType(qual_ty) => {
                        FuncOrQualType::Func(FuncType::new(qual_ty, func_args))
                    }
                    FuncOrQualType::Func(_) => panic!("bad"),
                };
            }
        }

        match func_or_qual_ty {
            FuncOrQualType::QualType(qual_ty) => Ok(Some(ExternalDecl::VarDecl(ident, qual_ty))),
            FuncOrQualType::Func(func) => Ok(Some(ExternalDecl::FuncDef(ident, func))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one_external_declaration(code: &str) -> Option<ExternalDecl> {
        let mut parser = match Parser::from_code(code) {
            Ok(parser) => parser,
            Err(err) => panic!(r#"Unexpected lexer error {:?} for "{:}""#, err, code),
        };
        let decl = match parser.read_next_external_declaration() {
            Ok(Some(decl)) => decl,
            Ok(None) => return None,
            Err(err) => panic!(r#"Unexpected error {:?} for "{:}""#, err, code),
        };
        match parser.read_next_external_declaration() {
            Ok(None) => Some(decl),
            Ok(Some(another)) => panic!(
                r#"Unexpected {:?} after {:?} for "{:}""#,
                another, decl, code
            ),
            Err(err) => panic!(
                r#"Unexpected error {:?} after {:?} for "{:}""#,
                err, decl, code
            ),
        }
    }

    fn ptr(to: QualType) -> QualType {
        QualType::new(Type::Pointer(Box::new(to)), TypeQualifiers::empty())
    }

    fn qual_ptr(to: QualType, qualifiers: TypeQualifiers) -> QualType {
        QualType::new(Type::Pointer(Box::new(to)), qualifiers)
    }

    fn func_ptr(ret_type: QualType, args: FuncArgs) -> QualType {
        QualType::new(
            Type::FuncPointer(Box::new(FuncType::new(ret_type, args))),
            TypeQualifiers::empty(),
        )
    }

    #[test]
    fn test_simple_declaration() {
        assert_eq!(parse_one_external_declaration(r#""#), None);
        assert_eq!(
            parse_one_external_declaration(r#"abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty())
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"(abcd);"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
            )),
        );
        assert_eq!(
            parse_one_external_declaration(r#"int abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"signed long;"#),
            Some(ExternalDecl::Nothing)
        );
        assert_eq!(
            parse_one_external_declaration(r#"unsigned char abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                QualType::new(
                    Type::Primitive(PrimitiveType::UnsignedChar),
                    TypeQualifiers::empty()
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"*abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                ptr(QualType::new(
                    Type::Primitive(PrimitiveType::Int),
                    TypeQualifiers::empty()
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"signed long long int * abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                ptr(QualType::new(
                    Type::Primitive(PrimitiveType::LongLong),
                    TypeQualifiers::empty()
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"const short * abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                ptr(QualType::new(
                    Type::Primitive(PrimitiveType::Short),
                    TypeQualifiers::CONST
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"short const * abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                ptr(QualType::new(
                    Type::Primitive(PrimitiveType::Short),
                    TypeQualifiers::CONST
                ))
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"float * const abcd;"#),
            Some(ExternalDecl::VarDecl(
                "abcd".to_string(),
                qual_ptr(
                    QualType::new(
                        Type::Primitive(PrimitiveType::Float),
                        TypeQualifiers::empty()
                    ),
                    TypeQualifiers::CONST
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"foo();"#),
            Some(ExternalDecl::FuncDef(
                "foo".to_string(),
                FuncType::new(
                    QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*foo)();"#),
            Some(ExternalDecl::VarDecl(
                "foo".to_string(),
                func_ptr(
                    QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*(foo))();"#),
            Some(ExternalDecl::VarDecl(
                "foo".to_string(),
                func_ptr(
                    QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*(*bar)())();"#),
            Some(ExternalDecl::VarDecl(
                "bar".to_string(),
                func_ptr(
                    func_ptr(
                        QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                        FuncArgs::Undefined
                    ),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"int (*foo())();"#),
            Some(ExternalDecl::FuncDef(
                "foo".to_string(),
                FuncType::new(
                    func_ptr(
                        QualType::new(Type::Primitive(PrimitiveType::Int), TypeQualifiers::empty()),
                        FuncArgs::Undefined
                    ),
                    FuncArgs::Undefined
                )
            ))
        );
        assert_eq!(
            parse_one_external_declaration(r#"char const * (**hogehoge)();"#),
            Some(ExternalDecl::VarDecl(
                "hogehoge".to_string(),
                ptr(func_ptr(
                    ptr(QualType::new(
                        Type::Primitive(PrimitiveType::Char),
                        TypeQualifiers::CONST
                    )),
                    FuncArgs::Undefined
                ))
            ))
        );
    }
}

fn main() -> Result<(), ParseError> {
    let mut parser = Parser::from_code(r#"x;"#)?;
    let decl = parser.read_next_external_declaration()?;
    println!("Declaration: {:?}", decl);
    Ok(())
}
