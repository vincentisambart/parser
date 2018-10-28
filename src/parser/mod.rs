use crate::error::{ParseError, ParseErrorKind};
use crate::failable::{FailableIterator, FailablePeekable};
use crate::lex::{Keyword, Literal, Position, PositionedToken, Punctuator, Token, TokenIter};

use bitflags::bitflags;
use std::collections::HashSet;

trait PeekingToken {
    type Error;

    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, Self::Error>;
    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, Self::Error>;
    fn next_if_any_ident(&mut self) -> Result<Option<(String, Position)>, Self::Error>;
    fn next_if_any_kw(&mut self) -> Result<Option<(Keyword, Position)>, Self::Error>;
}

impl<I> PeekingToken for FailablePeekable<I>
where
    I: FailableIterator<Item = PositionedToken>,
{
    type Error = I::Error;

    fn advance_if_kw(&mut self, kw: Keyword) -> Result<bool, Self::Error> {
        self.advance_if(|token| match token {
            PositionedToken(Token::Keyword(x), _) if *x == kw => true,
            _ => false,
        })
    }

    fn advance_if_punc(&mut self, punc: Punctuator) -> Result<bool, Self::Error> {
        self.advance_if(|token| match token {
            PositionedToken(Token::Punctuator(x), _) if *x == punc => true,
            _ => false,
        })
    }

    fn next_if_any_ident(&mut self) -> Result<Option<(String, Position)>, I::Error> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Identifier(_), _) => true,
            _ => false,
        })?;
        match token {
            Some(PositionedToken(Token::Identifier(ident), pos)) => Ok(Some((ident, pos))),
            Some(_) => unreachable!(),
            None => Ok(None),
        }
    }

    fn next_if_any_kw(&mut self) -> Result<Option<(Keyword, Position)>, I::Error> {
        let token = self.next_if(|token| match token {
            PositionedToken(Token::Keyword(_), _) => true,
            _ => false,
        })?;
        match token {
            Some(PositionedToken(Token::Keyword(kw), pos)) => Ok(Some((kw, pos))),
            Some(_) => unreachable!(),
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasicType {
    Void, // The standard doesn't consider "void" a basic type but that makes things easier
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    SignedInt, // Could be handled differently from Int in bit fields
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
        const RESTRICT = 1 << 2; // Only for pointers
        const ATOMIC   = 1 << 3;
    }
}

bitflags! {
    struct FuncSpecifiers: u8 {
        const INLINE   = 1;
        const NORETURN = 1 << 1;
    }
}

bitflags! {
    struct TypeKeywords: u16 {
        const VOID     = 1;
        const BOOL     = 1 << 1;
        const CHAR     = 1 << 2;
        const SHORT    = 1 << 3;
        const INT      = 1 << 4;
        const LONG_1   = 1 << 5;
        const LONG_2   = 1 << 6;
        const SIGNED   = 1 << 7;
        const UNSIGNED = 1 << 8;
        const DOUBLE   = 1 << 9;
        const FLOAT    = 1 << 10;
        const COMPLEX  = 1 << 11;

        const SIGNED_CHAR = Self::SIGNED.bits | Self::CHAR.bits;
        const UNSIGNED_CHAR = Self::UNSIGNED.bits | Self::CHAR.bits;
        const SHORT_INT = Self::SHORT.bits | Self::INT.bits;
        const SIGNED_SHORT = Self::SIGNED.bits | Self::SHORT.bits;
        const SIGNED_SHORT_INT = Self::SIGNED.bits | Self::SHORT.bits | Self::INT.bits;
        const UNSIGNED_SHORT = Self::UNSIGNED.bits | Self::SHORT.bits;
        const UNSIGNED_SHORT_INT = Self::UNSIGNED.bits | Self::SHORT.bits | Self::INT.bits;
        const SIGNED_INT = Self::SIGNED.bits | Self::INT.bits;
        const UNSIGNED_INT = Self::UNSIGNED.bits | Self::INT.bits;
        const LONG = Self::LONG_1.bits;
        const LONG_INT = Self::LONG_1.bits | Self::INT.bits;
        const SIGNED_LONG = Self::SIGNED.bits | Self::LONG_1.bits;
        const SIGNED_LONG_INT = Self::SIGNED.bits | Self::LONG_1.bits | Self::INT.bits;
        const UNSIGNED_LONG = Self::UNSIGNED.bits | Self::LONG_1.bits;
        const UNSIGNED_LONG_INT = Self::UNSIGNED.bits | Self::LONG_1.bits | Self::INT.bits;
        const LONG_LONG = Self::LONG_1.bits | Self::LONG_2.bits;
        const LONG_LONG_INT = Self::LONG_1.bits | Self::LONG_2.bits | Self::INT.bits;
        const SIGNED_LONG_LONG = Self::SIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits;
        const SIGNED_LONG_LONG_INT = Self::SIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits | Self::INT.bits;
        const UNSIGNED_LONG_LONG = Self::UNSIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits;
        const UNSIGNED_LONG_LONG_INT = Self::UNSIGNED.bits | Self::LONG_1.bits | Self::LONG_2.bits | Self::INT.bits;
        const LONG_DOUBLE = Self::LONG_1.bits | Self::DOUBLE.bits;
        const COMPLEX_FLOAT = Self::COMPLEX.bits | Self::FLOAT.bits;
        const COMPLEX_DOUBLE = Self::COMPLEX.bits | Self::DOUBLE.bits;
        const COMPLEX_LONG_DOUBLE = Self::COMPLEX.bits | Self::LONG_1.bits | Self::DOUBLE.bits;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Greater,
    GreaterEq,
    Less,
    LessEq,
    ShiftLeft,
    ShiftRight,
    BinaryOr,
    BinaryAnd,
    BinaryXor,
    LogicOr,
    LogicAnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Minus,
    Plus,
    Addr,
    Deref,
    BinNot,
    LogicNot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstExpr {
    Literal(Literal),
    Identifier(String),
    UnaryOp(UnaryOp, Box<ConstExpr>),
    BinaryOp(BinaryOp, Box<ConstExpr>, Box<ConstExpr>),
    TernaryOp(Box<ConstExpr>, Box<ConstExpr>, Box<ConstExpr>),
    SizeOfType(DerivedType),
    SizeOfExpr(Box<ConstExpr>),
    Cast(DerivedType, Box<ConstExpr>),
    // Some of the expressions below are by themselves not constant,
    // but can be given to sizeof in a constant expression.
    Subscript(Box<ConstExpr>, Box<ConstExpr>),
    Member(Box<ConstExpr>, Box<ConstExpr>),
    PtrMember(Box<ConstExpr>, Box<ConstExpr>),
    FuncCall(Box<ConstExpr>, Vec<ConstExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Initializer {
    Single(ConstExpr),
    List(Vec<Initializer>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumItem(String, Option<ConstExpr>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagItemDecl(QualifiedType, Vec<(String, Vec<Deriv>, Option<ConstExpr>)>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tag {
    Enum(Option<String>, Option<Vec<EnumItem>>),
    Struct(Option<String>, Option<Vec<TagItemDecl>>),
    Union(Option<String>, Option<Vec<TagItemDecl>>),
}

// impl Tag {
//     fn name(&self) -> &Option<String> {
//         match self {
//             Tag::Enum(name, _) => name,
//             Tag::Struct(name, _) => name,
//             Tag::Union(name, _) => name,
//         }
//     }
// }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnqualifiedType {
    Basic(BasicType),
    Tag(Tag),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedType(UnqualifiedType, TypeQualifiers);

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArraySize {
    Unspecified,
    Fixed(ConstExpr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedType(QualifiedType, Vec<Deriv>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncParam(Option<String>, Option<DerivedType>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnRFuncParamDecl(QualifiedType, Vec<(String, Vec<Deriv>)>);

#[derive(Debug, Clone, PartialEq, Eq)]
enum FuncDeclParams {
    Unspecified,
    Ansi {
        params: Vec<FuncParam>,
        is_variadic: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncDefParams {
    Unspecified,
    KnR {
        param_names: Vec<String>,
        decls: Vec<KnRFuncParamDecl>,
    },
    Ansi {
        params: Vec<FuncParam>,
        is_variadic: bool,
    },
}

impl From<FuncDeclParams> for FuncDefParams {
    fn from(params: FuncDeclParams) -> Self {
        match params {
            FuncDeclParams::Unspecified => FuncDefParams::Unspecified,
            FuncDeclParams::Ansi {
                params,
                is_variadic,
            } => FuncDefParams::Ansi {
                params,
                is_variadic,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Deriv {
    Ptr(TypeQualifiers),
    Func(FuncDeclParams),
    Array(ArraySize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDecl(QualifiedType, Vec<(String, Vec<Deriv>)>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl(
    Option<Linkage>,
    QualifiedType,
    Vec<(String, Vec<Deriv>, Option<Initializer>)>,
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncDef(
    String,
    Option<Linkage>,
    FuncSpecifiers,
    DerivedType,
    FuncDefParams,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Linkage {
    External,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtDecl {
    Decl(Decl),
    TypeDef(TypeDecl),
    FuncDef(FuncDef),
}

#[derive(Debug, Clone)]
enum DeclSpec {
    Decl {
        qual_type: QualifiedType,
        pos: Position,
        linkage: Option<Linkage>,
        func_specifiers: FuncSpecifiers,
        is_thread_local: bool,
    },
    TypeDef {
        qual_type: QualifiedType,
        pos: Position,
    },
}

struct TypeManager {
    types_stack: Vec<HashSet<String>>,
}

impl TypeManager {
    fn builtin_types() -> HashSet<String> {
        let mut builtins = HashSet::new();
        builtins.insert("__builtin_va_list".to_string());
        builtins
    }

    fn new() -> TypeManager {
        TypeManager {
            types_stack: vec![Self::builtin_types(), HashSet::new()],
        }
    }

    fn add_type_to_current_scope(&mut self, name: String) -> bool {
        self.types_stack.last_mut().unwrap().insert(name)
    }

    fn is_type_name(&self, name: &str) -> bool {
        self.types_stack
            .iter()
            .rev()
            .any(|types| types.contains(name))
    }
}

#[derive(Debug, Clone)]
struct DeclaratorParenLevel {
    ptr_qualifs: Vec<TypeQualifiers>,
    func_params: Option<FuncDeclParams>,
    array_sizes: Vec<ArraySize>,
}

impl DeclaratorParenLevel {
    fn new() -> DeclaratorParenLevel {
        DeclaratorParenLevel {
            ptr_qualifs: Vec::new(),
            func_params: None,
            array_sizes: Vec::new(),
        }
    }

    fn append_to(self, derivs: &mut Vec<Deriv>) {
        for qualifiers in self.ptr_qualifs {
            derivs.push(Deriv::Ptr(qualifiers));
        }
        if let Some(func_params) = self.func_params {
            derivs.push(Deriv::Func(func_params));
        }
        for array_size in self.array_sizes.into_iter().rev() {
            derivs.push(Deriv::Array(array_size));
        }
    }

    fn is_empty(&self) -> bool {
        self.ptr_qualifs.is_empty() && self.func_params.is_none() && self.array_sizes.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FuncDefParamsKind {
    Unspecified,
    KnR,
    Ansi,
}

impl FuncDefParamsKind {
    fn from_derivs(derivs: &[Deriv]) -> Option<FuncDefParamsKind> {
        if let Some(Deriv::Func(params)) = derivs.last() {
            Some(Self::from_params(params))
        } else {
            None
        }
    }

    fn from_params(params: &FuncDeclParams) -> FuncDefParamsKind {
        match params {
            FuncDeclParams::Ansi {
                params,
                is_variadic,
            } => {
                if !params.is_empty()
                    && !is_variadic
                    && params.iter().all(|param| {
                        let FuncParam(name, derivs) = param;
                        name.is_some() && derivs.is_none()
                    }) {
                    FuncDefParamsKind::KnR
                } else {
                    FuncDefParamsKind::Ansi
                }
            }
            FuncDeclParams::Unspecified => FuncDefParamsKind::Unspecified,
        }
    }
}

pub struct Parser<'a> {
    iter: FailablePeekable<TokenIter<'a>>,
    type_manager: TypeManager,
}

impl<'a> Parser<'a> {
    pub fn from_code(code: &'a str) -> Parser<'a> {
        let iter = TokenIter::from(code);
        Self::from(iter)
    }

    pub fn from(iter: TokenIter<'a>) -> Parser<'a> {
        Parser {
            iter: iter.peekable(),
            type_manager: TypeManager::new(),
        }
    }

    fn read_type_qualifier(&mut self) -> Result<Option<TypeQualifiers>, ParseError> {
        let qual = if self.iter.advance_if_kw(Keyword::Const)? {
            Some(TypeQualifiers::CONST)
        } else if self.iter.advance_if_kw(Keyword::Volatile)? {
            Some(TypeQualifiers::VOLATILE)
        } else if self.iter.advance_if_kw(Keyword::Restrict)? {
            Some(TypeQualifiers::RESTRICT)
        } else if self.iter.advance_if_kw(Keyword::Atomic)? {
            Some(TypeQualifiers::ATOMIC)
        } else {
            None
        };
        Ok(qual)
    }

    fn read_type_name(&mut self) -> Result<Option<(String, Position)>, ParseError> {
        let type_manager = &self.type_manager;
        if let Some(PositionedToken(Token::Identifier(ident), pos)) =
            self.iter.next_if(|token| match token {
                PositionedToken(Token::Identifier(ident), _) => {
                    type_manager.is_type_name(ident.as_ref())
                }
                _ => false,
            })? {
            Ok(Some((ident, pos)))
        } else {
            Ok(None)
        }
    }

    fn read_enum(&mut self, pos: &Position) -> Result<Tag, ParseError> {
        let enum_name = self.iter.next_if_any_ident()?.map(|(ident, _)| ident);
        let unqual_type = if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
            let mut enum_values = Vec::new();
            loop {
                if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                    break;
                } else if let Some((value_name, _)) = self.iter.next_if_any_ident()? {
                    if self.iter.advance_if_punc(Punctuator::Equal)? {
                        let value = self.read_const_expr()?;
                        enum_values.push(EnumItem(value_name, Some(value)));
                    } else {
                        enum_values.push(EnumItem(value_name, None));
                    }
                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        break;
                    } else {
                        self.expect_token(&Token::Punctuator(Punctuator::Comma))?;
                    }
                } else {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidConstruct,
                        "invalid enum declaration".to_string(),
                        pos.clone(),
                    ));
                }
            }
            Tag::Enum(enum_name, Some(enum_values))
        } else {
            if enum_name.is_none() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    "an enum declaration must have either a name or the list of values".to_string(),
                    pos.clone(),
                ));
            }
            Tag::Enum(enum_name, None)
        };
        Ok(unqual_type)
    }

    fn read_union_or_struct<F>(&mut self, pos: &Position, builder: F) -> Result<Tag, ParseError>
    where
        F: Fn(Option<String>, Option<Vec<TagItemDecl>>) -> Tag,
    {
        let struct_name = self.iter.next_if_any_ident()?.map(|(ident, _)| ident);
        let tag = if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
            let mut struct_field_decls = Vec::new();
            loop {
                if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                    break;
                }
                let root_type = match self.read_decl_spec()? {
                    Some(DeclSpec::TypeDef { pos, .. }) => {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a struct field cannot contain a typedef".to_string(),
                            pos,
                        ));
                    }
                    Some(DeclSpec::Decl {
                        qual_type,
                        pos,
                        linkage,
                        ..
                    }) => {
                        if linkage.is_some() {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a struct field cannot be extern or static".to_string(),
                                pos,
                            ));
                        }
                        qual_type
                    }
                    None => {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a struct field requires a type".to_string(),
                            pos.clone(),
                        ))
                    }
                };
                let mut fields = Vec::new();
                let mut did_finish_struct = false;
                loop {
                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        did_finish_struct = true;
                        break;
                    } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                        break;
                    }

                    let (ident, derivs) = self.read_declarator()?;
                    let (ident, _) = if let Some(ident) = ident {
                        ident
                    } else {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a struct field requires a name in most cases".to_string(),
                            pos.clone(),
                        ));
                    };

                    let bit_size = if self.iter.advance_if_punc(Punctuator::Colon)? {
                        Some(self.read_const_expr()?)
                    } else {
                        None
                    };

                    fields.push((ident, derivs, bit_size));

                    if self.iter.advance_if_punc(Punctuator::RightCurlyBracket)? {
                        did_finish_struct = true;
                        break;
                    } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                        break;
                    } else {
                        self.expect_token(&Token::Punctuator(Punctuator::Comma))?;
                    }
                }
                struct_field_decls.push(TagItemDecl(root_type, fields));
                if did_finish_struct {
                    break;
                }
            }
            builder(struct_name, Some(struct_field_decls))
        } else {
            if struct_name.is_none() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    "a struct declaration must have either a name or the list of fields"
                        .to_string(),
                    pos.clone(),
                ));
            }
            builder(struct_name, None)
        };
        Ok(tag)
    }

    fn expect_token(&mut self, expected: &Token) -> Result<Position, ParseError> {
        match self.iter.next()? {
            Some(PositionedToken(token, pos)) => {
                if token == *expected {
                    Ok(pos)
                } else {
                    let message = format!("expecting {:?}, got {:?}", expected, token);
                    Err(ParseError::new_with_position(
                        ParseErrorKind::UnexpectedToken(token),
                        message,
                        pos,
                    ))
                }
            }
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                format!(
                    "unfinished construct at end of line, expecting {:?}",
                    expected
                ),
            )),
        }
    }

    fn next_token_pos(&mut self) -> Result<Option<Position>, ParseError> {
        match self.iter.peek()? {
            Some(PositionedToken(_, pos)) => Ok(Some(pos.clone())),
            None => Ok(None),
        }
    }

    // Should be called just after having read an opening parenthesis.
    fn read_func_params(&mut self) -> Result<FuncDeclParams, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightParenthesis)? {
            // foo() means parameters are undefined
            return Ok(FuncDeclParams::Unspecified);
        }
        let mut params = Vec::new();
        let mut is_first_param = true;
        let mut is_variadic = false;
        loop {
            let root_type = match self.read_decl_spec()? {
                Some(DeclSpec::TypeDef { pos, .. }) => {
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::InvalidConstruct,
                        "a function parameter cannot contain a typedef".to_string(),
                        pos,
                    ));
                }
                Some(DeclSpec::Decl {
                    qual_type,
                    pos,
                    linkage,
                    ..
                }) => {
                    if linkage.is_some() {
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a function parameter cannot be extern or static".to_string(),
                            pos,
                        ));
                    }
                    Some(qual_type)
                }
                None => None,
            };
            if is_first_param {
                if let Some(QualifiedType(UnqualifiedType::Basic(BasicType::Void), qualifiers)) =
                    root_type
                {
                    if qualifiers.is_empty()
                        && self.iter.advance_if_punc(Punctuator::RightParenthesis)?
                    {
                        // foo(void) means no parameter
                        break;
                    }
                }
            }

            let (ident, derivs) = self.read_declarator()?;
            let param = if root_type.is_none() && derivs.is_empty() {
                // No type
                let (ident, _) = match ident {
                    Some(ident) => ident,
                    None => match self.next_token_pos()? {
                        Some(pos) => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a declaration should have at least an identifier or a type"
                                    .to_string(),
                                pos.clone(),
                            ))
                        }
                        None => {
                            return Err(ParseError::new(
                                ParseErrorKind::UnexpectedEOF,
                                "unfinished declaration at the end of the file".to_string(),
                            ))
                        }
                    },
                };
                FuncParam(Some(ident), None)
            } else {
                let root_type = root_type.unwrap_or_else(|| {
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty(),
                    )
                });
                FuncParam(
                    ident.map(|(ident, _)| ident),
                    Some(DerivedType(root_type, derivs)),
                )
            };
            params.push(param);

            is_first_param = false;
            if self.iter.advance_if_punc(Punctuator::Comma)? {
                if self.iter.advance_if_punc(Punctuator::Ellipsis)? {
                    is_variadic = true;
                    self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                    break;
                }
            } else {
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                break;
            }
        }

        Ok(FuncDeclParams::Ansi {
            params,
            is_variadic,
        })
    }

    fn read_primary_expr(&mut self) -> Result<ConstExpr, ParseError> {
        match self.iter.next()? {
            Some(PositionedToken(Token::Literal(literal), _)) => Ok(ConstExpr::Literal(literal)),
            Some(PositionedToken(Token::Identifier(ident), _)) => Ok(ConstExpr::Identifier(ident)),
            Some(PositionedToken(Token::Punctuator(Punctuator::LeftParenthesis), _)) => {
                let expr = self.read_const_expr()?;
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                Ok(expr)
            }
            Some(PositionedToken(token, position)) => Err(ParseError::new_with_position(
                ParseErrorKind::UnexpectedToken(token),
                "currently only supporting integer literal constant values".to_string(),
                position,
            )),
            None => Err(ParseError::new(
                ParseErrorKind::UnexpectedEOF,
                "end of line when expecting an expression value".to_string(),
            )),
        }
    }

    fn read_postfix_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let expr = self.read_primary_expr()?;
        loop {
            if self.iter.advance_if_punc(Punctuator::LeftSquareBracket)? {
                unimplemented!()
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                unimplemented!()
            } else if self.iter.advance_if_punc(Punctuator::Period)? {
                unimplemented!()
            } else if self.iter.advance_if_punc(Punctuator::Arrow)? {
                unimplemented!()
            } else {
                break Ok(expr);
            }
        }
    }

    fn read_unary_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let unary_expr = if self.iter.advance_if_punc(Punctuator::Ampersand)? {
            let expr = self.read_cast_expr()?;
            ConstExpr::UnaryOp(UnaryOp::Addr, Box::new(expr))
        } else if self.iter.advance_if_punc(Punctuator::Star)? {
            let expr = self.read_cast_expr()?;
            ConstExpr::UnaryOp(UnaryOp::Deref, Box::new(expr))
        } else if self.iter.advance_if_punc(Punctuator::Plus)? {
            let expr = self.read_cast_expr()?;
            ConstExpr::UnaryOp(UnaryOp::Plus, Box::new(expr))
        } else if self.iter.advance_if_punc(Punctuator::Minus)? {
            let expr = self.read_cast_expr()?;
            ConstExpr::UnaryOp(UnaryOp::Minus, Box::new(expr))
        } else if self.iter.advance_if_punc(Punctuator::Tilde)? {
            let expr = self.read_cast_expr()?;
            ConstExpr::UnaryOp(UnaryOp::BinNot, Box::new(expr))
        } else if self.iter.advance_if_punc(Punctuator::Exclamation)? {
            let expr = self.read_cast_expr()?;
            ConstExpr::UnaryOp(UnaryOp::LogicNot, Box::new(expr))
        } else if self.iter.advance_if_kw(Keyword::Sizeof)? {
            if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                if self.is_before_type()? {
                    let derived_type = self.read_just_type()?;
                    self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                    ConstExpr::SizeOfType(derived_type)
                } else {
                    let expr = self.read_const_expr()?;
                    self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                    ConstExpr::SizeOfExpr(Box::new(expr))
                }
            } else {
                let expr = self.read_unary_expr()?;
                ConstExpr::SizeOfExpr(Box::new(expr))
            }
        } else {
            self.read_postfix_expr()?
        };
        Ok(unary_expr)
    }

    // Read a type without variable name, for casts or sizeof.
    // Must only be called if you already know there's a type to read.
    fn read_just_type(&mut self) -> Result<DerivedType, ParseError> {
        let qual_type = match self.read_decl_spec()? {
            Some(DeclSpec::TypeDef { pos, .. }) => {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    "expection a type, not a typedef".to_string(),
                    pos,
                ));
            }
            Some(DeclSpec::Decl { qual_type, .. }) => qual_type,
            None => unreachable!(),
        };
        let (ident, derivs) = self.read_declarator()?;
        if let Some((_, pos)) = ident {
            return Err(ParseError::new_with_position(
                ParseErrorKind::InvalidConstruct,
                "expection a type without variable name".to_string(),
                pos,
            ));
        }
        Ok(DerivedType(qual_type, derivs))
    }

    fn is_before_type(&mut self) -> Result<bool, ParseError> {
        let looks_like_type = match self.iter.peek()? {
            Some(PositionedToken(Token::Keyword(kw), _)) => match kw {
                Keyword::Const
                | Keyword::Volatile
                | Keyword::Restrict
                | Keyword::Atomic
                | Keyword::Void
                | Keyword::Int
                | Keyword::Long
                | Keyword::Short
                | Keyword::Char
                | Keyword::Signed
                | Keyword::Unsigned
                | Keyword::Double
                | Keyword::Float
                | Keyword::Bool
                | Keyword::Complex
                | Keyword::Enum
                | Keyword::Struct
                | Keyword::Union => true,
                _ => false,
            },
            Some(PositionedToken(Token::Identifier(ident), _)) => {
                self.type_manager.is_type_name(ident.as_ref())
            }
            _ => false,
        };
        Ok(looks_like_type)
    }

    fn read_cast_expr(&mut self) -> Result<ConstExpr, ParseError> {
        if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
            // Is it a cast, or a parenthesized expression?
            if self.is_before_type()? {
                let derived_type = self.read_just_type()?;
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                let expr = self.read_cast_expr()?;
                Ok(ConstExpr::Cast(derived_type, Box::new(expr)))
            } else {
                let expr = self.read_const_expr()?;
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
                Ok(expr)
            }
        } else {
            self.read_unary_expr()
        }
    }

    fn read_mul_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_cast_expr()?;
        loop {
            if self.iter.advance_if_punc(Punctuator::Star)? {
                let rhs = self.read_cast_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Mul, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::Slash)? {
                let rhs = self.read_cast_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Div, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::Percent)? {
                let rhs = self.read_cast_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Mod, Box::new(expr), Box::new(rhs));
            } else {
                break Ok(expr);
            }
        }
    }

    fn read_add_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_mul_expr()?;
        loop {
            if self.iter.advance_if_punc(Punctuator::Plus)? {
                let rhs = self.read_mul_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Add, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::Minus)? {
                let rhs = self.read_mul_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Sub, Box::new(expr), Box::new(rhs));
            } else {
                break Ok(expr);
            }
        }
    }

    fn read_shift_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_add_expr()?;
        loop {
            if self.iter.advance_if_punc(Punctuator::LessLess)? {
                let rhs = self.read_add_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::ShiftLeft, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::GreaterGreater)? {
                let rhs = self.read_add_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::ShiftRight, Box::new(expr), Box::new(rhs));
            } else {
                break Ok(expr);
            }
        }
    }

    fn read_rel_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_shift_expr()?;
        loop {
            if self.iter.advance_if_punc(Punctuator::Less)? {
                let rhs = self.read_shift_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Less, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::Greater)? {
                let rhs = self.read_shift_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Greater, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::LessEqual)? {
                let rhs = self.read_shift_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::LessEq, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::GreaterEqual)? {
                let rhs = self.read_shift_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::GreaterEq, Box::new(expr), Box::new(rhs));
            } else {
                break Ok(expr);
            }
        }
    }

    fn read_eq_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_rel_expr()?;
        loop {
            if self.iter.advance_if_punc(Punctuator::EqualEqual)? {
                let rhs = self.read_rel_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::Eq, Box::new(expr), Box::new(rhs));
            } else if self.iter.advance_if_punc(Punctuator::ExclamationEqual)? {
                let rhs = self.read_rel_expr()?;
                expr = ConstExpr::BinaryOp(BinaryOp::NotEq, Box::new(expr), Box::new(rhs));
            } else {
                break Ok(expr);
            }
        }
    }

    fn read_and_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_eq_expr()?;
        while self.iter.advance_if_punc(Punctuator::Ampersand)? {
            let rhs = self.read_eq_expr()?;
            expr = ConstExpr::BinaryOp(BinaryOp::BinaryAnd, Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn read_xor_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_and_expr()?;
        while self.iter.advance_if_punc(Punctuator::Caret)? {
            let rhs = self.read_and_expr()?;
            expr = ConstExpr::BinaryOp(BinaryOp::BinaryXor, Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn read_or_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_xor_expr()?;
        while self.iter.advance_if_punc(Punctuator::Pipe)? {
            let rhs = self.read_xor_expr()?;
            expr = ConstExpr::BinaryOp(BinaryOp::BinaryOr, Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn read_logical_and_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_or_expr()?;
        while self.iter.advance_if_punc(Punctuator::AmpersandAmpersand)? {
            let rhs = self.read_or_expr()?;
            expr = ConstExpr::BinaryOp(BinaryOp::LogicAnd, Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn read_logical_or_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let mut expr = self.read_logical_and_expr()?;
        while self.iter.advance_if_punc(Punctuator::PipePipe)? {
            let rhs = self.read_logical_and_expr()?;
            expr = ConstExpr::BinaryOp(BinaryOp::LogicOr, Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    fn read_const_expr(&mut self) -> Result<ConstExpr, ParseError> {
        let expr = self.read_logical_or_expr()?;
        if self.iter.advance_if_punc(Punctuator::Question)? {
            let second_expr = self.read_const_expr()?;
            self.expect_token(&Token::Punctuator(Punctuator::Colon))?;
            let third_expr = self.read_const_expr()?;
            Ok(ConstExpr::TernaryOp(
                Box::new(expr),
                Box::new(second_expr),
                Box::new(third_expr),
            ))
        } else {
            Ok(expr)
        }
    }

    // Should be called just after having read an opening square bracket.
    fn read_array_size(&mut self) -> Result<ArraySize, ParseError> {
        if self.iter.advance_if_punc(Punctuator::RightSquareBracket)? {
            return Ok(ArraySize::Unspecified);
        }

        let size = self.read_const_expr()?;
        self.expect_token(&Token::Punctuator(Punctuator::RightSquareBracket))?;
        Ok(ArraySize::Fixed(size))
    }

    fn read_declarator(&mut self) -> Result<(Option<(String, Position)>, Vec<Deriv>), ParseError> {
        let mut levels = Vec::new();
        let mut current_level = DeclaratorParenLevel::new();
        let mut func_params_paren_already_opened = false;
        let ident = loop {
            if self.iter.advance_if_punc(Punctuator::Star)? {
                let mut qualifiers = TypeQualifiers::empty();
                while let Some(qualifier) = self.read_type_qualifier()? {
                    qualifiers |= qualifier;
                }
                current_level.ptr_qualifs.push(qualifiers);
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                levels.push(current_level);
                current_level = DeclaratorParenLevel::new();
            } else {
                levels.push(current_level);
                match self.iter.peek()? {
                    Some(PositionedToken(Token::Identifier(ident), _)) => {
                        if self.type_manager.is_type_name(ident)
                            && levels.len() > 1 /* at least one parenthesis */
                            && levels.iter().all(|level| level.is_empty())
                        {
                            // a type name just after one or more parentheses (and nothing else) is the first parameter of a function definition
                            func_params_paren_already_opened = true;
                            levels.pop();
                            break None;
                        } else {
                            break self.iter.next_if_any_ident()?;
                        }
                    }
                    _ => break None,
                }
            }
        };
        for (i, level) in levels.iter_mut().enumerate().rev() {
            if func_params_paren_already_opened {
                level.func_params = Some(self.read_func_params()?);
                func_params_paren_already_opened = false;
            } else if self.iter.advance_if_punc(Punctuator::LeftParenthesis)? {
                level.func_params = Some(self.read_func_params()?);
            }
            while self.iter.advance_if_punc(Punctuator::LeftSquareBracket)? {
                level.array_sizes.push(self.read_array_size()?);
            }
            if i != 0 {
                self.expect_token(&Token::Punctuator(Punctuator::RightParenthesis))?;
            }
        }
        let mut derivs = Vec::new();
        for level in levels {
            level.append_to(&mut derivs);
        }
        Ok((ident, derivs))
    }

    fn make_basic_type(type_kws: TypeKeywords, pos: &Position) -> Result<BasicType, ParseError> {
        let basic_type = match type_kws {
            TypeKeywords::VOID => BasicType::Void,
            TypeKeywords::BOOL => BasicType::Bool,
            TypeKeywords::CHAR => BasicType::Char,
            TypeKeywords::SIGNED_CHAR => BasicType::SignedChar,
            TypeKeywords::UNSIGNED_CHAR => BasicType::UnsignedChar,
            TypeKeywords::SHORT
            | TypeKeywords::SHORT_INT
            | TypeKeywords::SIGNED_SHORT
            | TypeKeywords::SIGNED_SHORT_INT => BasicType::Short,
            TypeKeywords::UNSIGNED_SHORT | TypeKeywords::UNSIGNED_SHORT_INT => {
                BasicType::UnsignedShort
            }
            TypeKeywords::INT => BasicType::Int,
            TypeKeywords::SIGNED | TypeKeywords::SIGNED_INT => BasicType::SignedInt,
            TypeKeywords::UNSIGNED | TypeKeywords::UNSIGNED_INT => BasicType::UnsignedInt,
            TypeKeywords::LONG
            | TypeKeywords::LONG_INT
            | TypeKeywords::SIGNED_LONG
            | TypeKeywords::SIGNED_LONG_INT => BasicType::Long,
            TypeKeywords::UNSIGNED_LONG | TypeKeywords::UNSIGNED_LONG_INT => {
                BasicType::UnsignedLong
            }
            TypeKeywords::LONG_LONG
            | TypeKeywords::LONG_LONG_INT
            | TypeKeywords::SIGNED_LONG_LONG
            | TypeKeywords::SIGNED_LONG_LONG_INT => BasicType::LongLong,
            TypeKeywords::UNSIGNED_LONG_LONG | TypeKeywords::UNSIGNED_LONG_LONG_INT => {
                BasicType::UnsignedLongLong
            }
            TypeKeywords::DOUBLE => BasicType::Double,
            TypeKeywords::FLOAT => BasicType::Float,
            TypeKeywords::COMPLEX => BasicType::DoubleComplex,
            TypeKeywords::LONG_DOUBLE => BasicType::LongDouble,
            TypeKeywords::COMPLEX_FLOAT => BasicType::FloatComplex,
            TypeKeywords::COMPLEX_DOUBLE => BasicType::DoubleComplex,
            TypeKeywords::COMPLEX_LONG_DOUBLE => BasicType::LongDoubleComplex,
            _ => {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "invalid combination of basic type keywords".to_string(),
                    pos.clone(),
                ))
            }
        };
        Ok(basic_type)
    }

    fn read_decl_spec(&mut self) -> Result<Option<DeclSpec>, ParseError> {
        let mut type_kws = TypeKeywords::empty();
        let mut decl_pos = None;
        let mut type_name = None;
        let mut type_qual = TypeQualifiers::empty();
        let mut tag = None;
        let mut is_typedef = false;
        let mut has_useless_kw = false;
        let mut linkage = None;
        let mut is_thread_local = false;
        let mut func_specifiers = FuncSpecifiers::empty();
        loop {
            match self.iter.next_if_any_kw()? {
                Some((kw, pos)) => {
                    match kw {
                        Keyword::Typedef => is_typedef = true,
                        Keyword::Const => type_qual |= TypeQualifiers::CONST,
                        Keyword::Volatile => type_qual |= TypeQualifiers::CONST,
                        Keyword::Restrict => type_qual |= TypeQualifiers::CONST,
                        Keyword::Atomic => type_qual |= TypeQualifiers::CONST,
                        // auto and register don't mean anything anymore
                        Keyword::Auto | Keyword::Register => has_useless_kw = true,
                        Keyword::Void => type_kws |= TypeKeywords::VOID,
                        Keyword::Int => type_kws |= TypeKeywords::INT,
                        Keyword::Long => {
                            // When an expression has multiple "long", the first one is LONG_1, and the other ones are LONG_2
                            if type_kws.contains(TypeKeywords::LONG_1) {
                                type_kws |= TypeKeywords::LONG_2
                            } else {
                                type_kws |= TypeKeywords::LONG_1
                            }
                        }
                        Keyword::Short => type_kws |= TypeKeywords::SHORT,
                        Keyword::Char => type_kws |= TypeKeywords::CHAR,
                        Keyword::Signed => type_kws |= TypeKeywords::SIGNED,
                        Keyword::Unsigned => type_kws |= TypeKeywords::UNSIGNED,
                        Keyword::Double => type_kws |= TypeKeywords::DOUBLE,
                        Keyword::Float => type_kws |= TypeKeywords::FLOAT,
                        Keyword::Bool => type_kws |= TypeKeywords::BOOL,
                        Keyword::Complex => type_kws |= TypeKeywords::COMPLEX,
                        Keyword::Enum | Keyword::Struct | Keyword::Union => {
                            if tag.is_some() {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration can only have one tag declaration".to_string(),
                                    pos,
                                ));
                            }
                            tag = Some(match kw {
                                Keyword::Enum => self.read_enum(&pos)?,
                                Keyword::Struct => self.read_union_or_struct(&pos, Tag::Struct)?,
                                Keyword::Union => self.read_union_or_struct(&pos, Tag::Union)?,
                                _ => unreachable!(),
                            });
                        }
                        Keyword::Extern => {
                            if linkage == Some(Linkage::Internal) {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration cannot be both extern and static".to_string(),
                                    pos,
                                ));
                            }
                            linkage = Some(Linkage::External);
                        }
                        Keyword::Static => {
                            if linkage == Some(Linkage::External) {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a declaration cannot be both extern and static".to_string(),
                                    pos,
                                ));
                            }
                            linkage = Some(Linkage::Internal);
                        }
                        Keyword::Noreturn => func_specifiers |= FuncSpecifiers::NORETURN,
                        Keyword::Inline => func_specifiers |= FuncSpecifiers::INLINE,
                        Keyword::ThreadLocal => is_thread_local = true,
                        kw => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                format!("unexpected keyword {:?} in declaration", kw),
                                pos,
                            ))
                        }
                    };
                    decl_pos = Some(pos.clone());
                }
                None => match self.read_type_name()? {
                    Some((ident, pos)) => {
                        decl_pos = Some(pos);
                        type_name = Some(ident);
                    }
                    None => break,
                },
            }
        }
        let decl_pos = match decl_pos {
            Some(decl_pos) => decl_pos,
            None => {
                assert!(
                    !is_typedef
                        && type_kws.is_empty()
                        && linkage.is_none()
                        && tag.is_none()
                        && !has_useless_kw
                );
                return Ok(None);
            }
        };
        let unqual_type = if let Some(type_name) = type_name {
            if !type_kws.is_empty() || tag.is_some() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    format!(
                        "a declaration cannot have both a basic type and a custom type {}",
                        type_name
                    ),
                    decl_pos,
                ));
            }
            UnqualifiedType::Custom(type_name)
        } else if let Some(tag) = tag {
            if !type_kws.is_empty() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a declaration cannot have both a basic type and a tag type".to_string(),
                    decl_pos,
                ));
            }
            UnqualifiedType::Tag(tag)
        } else if type_kws.is_empty() {
            UnqualifiedType::Basic(BasicType::Int)
        } else {
            let basic_type = Self::make_basic_type(type_kws, &decl_pos)?;
            UnqualifiedType::Basic(basic_type)
        };
        let qual_type = QualifiedType(unqual_type, type_qual);
        let decl_spec = if is_typedef {
            if linkage.is_some() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a typedef cannot use static or extern".to_string(),
                    decl_pos,
                ));
            }
            if !func_specifiers.is_empty() {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a typedef cannot use inline or _Noreturn".to_string(),
                    decl_pos,
                ));
            }
            if is_thread_local {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidType,
                    "a typedef cannot use _Thread_local".to_string(),
                    decl_pos,
                ));
            }
            DeclSpec::TypeDef {
                qual_type,
                pos: decl_pos,
            }
        } else {
            DeclSpec::Decl {
                qual_type,
                linkage,
                pos: decl_pos,
                func_specifiers,
                is_thread_local,
            }
        };
        Ok(Some(decl_spec))
    }

    // Should be called just after the opening curly brace of the function body has been read.
    fn skip_func_body(&mut self) -> Result<(), ParseError> {
        let mut opened_curlies = 1;
        loop {
            match self.iter.next()? {
                Some(PositionedToken(Token::Punctuator(Punctuator::LeftCurlyBracket), _)) => {
                    opened_curlies += 1
                }
                Some(PositionedToken(Token::Punctuator(Punctuator::RightCurlyBracket), _)) => {
                    opened_curlies -= 1;
                    if opened_curlies == 0 {
                        break;
                    }
                }
                None => {
                    return Err(ParseError::new(
                        ParseErrorKind::UnexpectedEOF,
                        format!(
                            "end of file even though {} curly braces were not closed",
                            opened_curlies
                        ),
                    ))
                }
                _ => (),
            }
        }
        Ok(())
    }

    fn read_func_def(
        &mut self,
        ident: String,
        pos: Position,
        decl_spec: Option<DeclSpec>,
        mut derivs: Vec<Deriv>,
    ) -> Result<ExtDecl, ParseError> {
        let params = if let Deriv::Func(params) = derivs.pop().unwrap() {
            params
        } else {
            unreachable!()
        };

        let (qual_type, linkage, func_specifiers, is_thread_local) = match decl_spec {
            Some(DeclSpec::Decl {
                qual_type,
                linkage,
                func_specifiers,
                is_thread_local,
                ..
            }) => (qual_type, linkage, func_specifiers, is_thread_local),
            None => (
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty(),
                ),
                None,
                FuncSpecifiers::empty(),
                false,
            ),
            _ => unreachable!(),
        };

        if is_thread_local {
            return Err(ParseError::new_with_position(
                ParseErrorKind::InvalidConstruct,
                "only variable declaration can be thread local".to_string(),
                pos,
            ));
        }

        let params = match FuncDefParamsKind::from_params(&params) {
            FuncDefParamsKind::Unspecified | FuncDefParamsKind::Ansi => {
                self.expect_token(&Token::Punctuator(Punctuator::LeftCurlyBracket))?;
                params.into()
            }
            FuncDefParamsKind::KnR => {
                let mut param_decls = Vec::new();
                loop {
                    if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
                        break;
                    }
                    let root_type = match self.read_decl_spec()? {
                        Some(DeclSpec::TypeDef { pos, .. }) => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a function parameter cannot be a typedef".to_string(),
                                pos,
                            ));
                        }
                        Some(DeclSpec::Decl {
                            qual_type,
                            pos,
                            linkage,
                            ..
                        }) => {
                            if linkage.is_some() {
                                return Err(ParseError::new_with_position(
                                    ParseErrorKind::InvalidConstruct,
                                    "a function parameter cannot be extern or static".to_string(),
                                    pos,
                                ));
                            }
                            qual_type
                        }
                        None => {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a function parameter requires a type".to_string(),
                                pos.clone(),
                            ))
                        }
                    };
                    let mut fields = Vec::new();
                    let mut did_finish_params_decl = false;
                    loop {
                        if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
                            did_finish_params_decl = true;
                            break;
                        } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                            break;
                        }

                        let (ident, derivs) = self.read_declarator()?;
                        let (ident, _) = if let Some(ident) = ident {
                            ident
                        } else {
                            return Err(ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a struct field requires a name in most cases".to_string(),
                                pos.clone(),
                            ));
                        };

                        fields.push((ident, derivs));

                        if self.iter.advance_if_punc(Punctuator::LeftCurlyBracket)? {
                            did_finish_params_decl = true;
                            break;
                        } else if self.iter.advance_if_punc(Punctuator::Semicolon)? {
                            break;
                        } else {
                            self.expect_token(&Token::Punctuator(Punctuator::Comma))?;
                        }
                    }
                    param_decls.push(KnRFuncParamDecl(root_type, fields));
                    if did_finish_params_decl {
                        break;
                    }
                }

                let params = match params {
                    FuncDeclParams::Ansi {
                        params,
                        is_variadic,
                    } => {
                        assert!(!is_variadic);
                        params
                    }
                    _ => unreachable!(),
                };
                let param_names = params
                    .into_iter()
                    .map(|param| match param {
                        FuncParam(Some(ident), None) => ident,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();

                FuncDefParams::KnR {
                    param_names,
                    decls: param_decls,
                }
            }
        };
        self.skip_func_body()?;
        let def = FuncDef(
            ident,
            linkage,
            func_specifiers,
            DerivedType(qual_type, derivs),
            params,
        );
        Ok(ExtDecl::FuncDef(def))
    }

    fn read_type_def(&mut self, qual_type: QualifiedType) -> Result<ExtDecl, ParseError> {
        if self.iter.advance_if_punc(Punctuator::Semicolon)? {
            return Ok(ExtDecl::TypeDef(TypeDecl(qual_type, Vec::new())));
        }

        let mut declarators = Vec::new();
        loop {
            let (ident, derivs) = self.read_declarator()?;
            let (ident, pos) = match ident {
                Some(ident) => ident,
                None => {
                    return Err(match self.next_token_pos()? {
                        Some(pos) => ParseError::new_with_position(
                            ParseErrorKind::InvalidConstruct,
                            "a typedef should have an identifier".to_string(),
                            pos,
                        ),
                        None => ParseError::new(
                            ParseErrorKind::UnexpectedEOF,
                            "unfinished typedef at the end of the file".to_string(),
                        ),
                    })
                }
            };

            if !self.type_manager.add_type_to_current_scope(ident.clone()) {
                return Err(ParseError::new_with_position(
                    ParseErrorKind::InvalidConstruct,
                    format!("trying to redefine the already defined type {}", ident),
                    pos,
                ));
            }

            declarators.push((ident, derivs));

            match self.iter.next()? {
                Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _)) => break,
                Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => (),
                Some(PositionedToken(token, pos)) => {
                    let message = format!(
                        "got {:?} in typedef where expecting a comma or semicolon",
                        token
                    );
                    return Err(ParseError::new_with_position(
                        ParseErrorKind::UnexpectedToken(token),
                        message,
                        pos,
                    ));
                }
                None => {
                    return Err(ParseError::new(
                        ParseErrorKind::UnexpectedEOF,
                        "end of file where expecting a comma or semicolon".to_string(),
                    ))
                }
            }
        }

        Ok(ExtDecl::TypeDef(TypeDecl(qual_type, declarators)))
    }

    fn read_decl_or_def(&mut self, decl_spec: Option<DeclSpec>) -> Result<ExtDecl, ParseError> {
        let mut declarators = Vec::new();
        if !self.iter.advance_if_punc(Punctuator::Semicolon)? {
            loop {
                let (ident, derivs) = self.read_declarator()?;
                let (ident, pos) = match ident {
                    Some(ident) => ident,
                    None => {
                        return Err(match self.next_token_pos()? {
                            Some(pos) => ParseError::new_with_position(
                                ParseErrorKind::InvalidConstruct,
                                "a declaration should have an identifier".to_string(),
                                pos,
                            ),
                            None => ParseError::new(
                                ParseErrorKind::UnexpectedEOF,
                                "unfinished declaration at the end of the file".to_string(),
                            ),
                        })
                    }
                };

                let is_func_def = if declarators.is_empty()
                    && FuncDefParamsKind::from_derivs(&derivs).is_some()
                {
                    match self.iter.peek()? {
                        // If the next token is a ";" or ",", we're in a function declaration, not definition
                        Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _))
                        | Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => false,
                        _ => true,
                    }
                } else {
                    false
                };

                if is_func_def {
                    return Ok(self.read_func_def(ident, pos, decl_spec, derivs)?);
                }

                let const_expr = if self.iter.advance_if_punc(Punctuator::Equal)? {
                    // TODO: Could be an initializer list {1, 2, 3}
                    Some(Initializer::Single(self.read_const_expr()?))
                } else {
                    None
                };

                declarators.push((ident, derivs, const_expr));

                match self.iter.next()? {
                    Some(PositionedToken(Token::Punctuator(Punctuator::Semicolon), _)) => break,
                    Some(PositionedToken(Token::Punctuator(Punctuator::Comma), _)) => (),
                    Some(PositionedToken(token, pos)) => {
                        let message = format!(
                            "got {:?} in declaration where expecting a comma or semicolon",
                            token
                        );
                        return Err(ParseError::new_with_position(
                            ParseErrorKind::UnexpectedToken(token),
                            message,
                            pos,
                        ));
                    }
                    None => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedEOF,
                            "end of file where expecting a comma or semicolon".to_string(),
                        ))
                    }
                }
            }
        }

        let ext_decl = match decl_spec {
            Some(DeclSpec::TypeDef { .. }) => {
                panic!("read_decl_or_def should not be called with a typedef")
            }
            Some(DeclSpec::Decl {
                qual_type, linkage, ..
            }) => ExtDecl::Decl(Decl(linkage, qual_type, declarators)),
            None => ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty(),
                ),
                declarators,
            )),
        };
        Ok(ext_decl)
    }
}

impl<'a> FailableIterator for Parser<'a> {
    type Item = ExtDecl;
    type Error = ParseError;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.iter.peek()?.is_none() {
            return Ok(None);
        }
        let decl_spec = self.read_decl_spec()?;
        let ext_decl = match decl_spec {
            Some(DeclSpec::TypeDef { qual_type, .. }) => self.read_type_def(qual_type)?,
            _ => self.read_decl_or_def(decl_spec)?,
        };
        Ok(Some(ext_decl))
    }
}

#[cfg(test)]
mod tests;
