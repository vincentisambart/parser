use super::*;

fn parse_external_declarations(code: &str) -> Vec<ExtDecl> {
    let parser = Parser::from_code(code);
    match parser.collect() {
        Ok(decls) => decls,
        Err(err) => panic!(r#"Unexpected error {:?} for "{:}""#, err, code),
    }
}

#[test]
fn test_simple_declaration() {
    assert_eq!(parse_external_declarations(r#""#), vec![]);
    assert_eq!(
        parse_external_declarations(r#"abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![("abcd".to_string(), vec![], None)]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"(abcd);"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![("abcd".to_string(), vec![], None)]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"int abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![("abcd".to_string(), vec![], None)]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"signed long;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Long),
                TypeQualifiers::empty()
            ),
            vec![]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"unsigned char abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::UnsignedChar),
                TypeQualifiers::empty()
            ),
            vec![("abcd".to_string(), vec![], None)]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"*abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "abcd".to_string(),
                vec![Deriv::Ptr(TypeQualifiers::empty())],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"signed long long int * abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::LongLong),
                TypeQualifiers::empty()
            ),
            vec![(
                "abcd".to_string(),
                vec![Deriv::Ptr(TypeQualifiers::empty())],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"const short * abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Short),
                TypeQualifiers::CONST
            ),
            vec![(
                "abcd".to_string(),
                vec![Deriv::Ptr(TypeQualifiers::empty())],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"short const * abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Short),
                TypeQualifiers::CONST
            ),
            vec![(
                "abcd".to_string(),
                vec![Deriv::Ptr(TypeQualifiers::empty())],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"float * const abcd;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Float),
                TypeQualifiers::empty()
            ),
            vec![(
                "abcd".to_string(),
                vec![Deriv::Ptr(TypeQualifiers::CONST)],
                None
            )]
        ))]
    );
}

#[test]
fn test_function_declaration() {
    assert_eq!(
        parse_external_declarations(r#"foo();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![Deriv::Func(FuncDeclParams::Unspecified)],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(void);"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Void),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![],
                    is_variadic: false
                })],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(int a, char b, ...);"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Void),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![
                        FuncParam(
                            Some("a".to_string()),
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Basic(BasicType::Int),
                                    TypeQualifiers::empty()
                                ),
                                vec![]
                            ))
                        ),
                        FuncParam(
                            Some("b".to_string()),
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Basic(BasicType::Char),
                                    TypeQualifiers::empty()
                                ),
                                vec![]
                            ))
                        )
                    ],
                    is_variadic: true
                })],
                None
            )]
        ))]
    );
    // function returning a function pointer
    assert_eq!(
        parse_external_declarations(r#"short (*foo())();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Short),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Func(FuncDeclParams::Unspecified)
                ],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(int(x));"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Void),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![FuncParam(
                        Some("x".to_string()),
                        Some(DerivedType(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Int),
                                TypeQualifiers::empty()
                            ),
                            vec![]
                        ))
                    )],
                    is_variadic: false
                })],
                None
            )]
        ))]
    );
    // Yes, C types declaration is crazy stuff
    assert_eq!(
        parse_external_declarations(r#"typedef int x; void foo(int(x));"#),
        vec![
            ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("x".to_string(), vec![])]
            )),
            ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "foo".to_string(),
                    vec![Deriv::Func(FuncDeclParams::Ansi {
                        params: vec![FuncParam(
                            None,
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Basic(BasicType::Int),
                                    TypeQualifiers::empty()
                                ),
                                vec![Deriv::Func(FuncDeclParams::Ansi {
                                    params: vec![FuncParam(
                                        None,
                                        Some(DerivedType(
                                            QualifiedType(
                                                UnqualifiedType::Custom("x".to_string()),
                                                TypeQualifiers::empty()
                                            ),
                                            vec![]
                                        ))
                                    )],
                                    is_variadic: false
                                })]
                            ))
                        )],
                        is_variadic: false
                    })],
                    None
                )]
            ))
        ]
    );
    assert_eq!(
        parse_external_declarations(r#"typedef int x; void foo(x);"#),
        vec![
            ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("x".to_string(), vec![])]
            )),
            ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "foo".to_string(),
                    vec![Deriv::Func(FuncDeclParams::Ansi {
                        params: vec![FuncParam(
                            None,
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Custom("x".to_string()),
                                    TypeQualifiers::empty()
                                ),
                                vec![]
                            ))
                        )],
                        is_variadic: false
                    })],
                    None
                )]
            ))
        ]
    );
    // That C compilers allow that is beyond me.
    assert_eq!(
        parse_external_declarations(r#"typedef int x; void foo(int((x, char)));"#),
        vec![
            ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("x".to_string(), vec![])]
            )),
            ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![(
                    "foo".to_string(),
                    vec![Deriv::Func(FuncDeclParams::Ansi {
                        params: vec![FuncParam(
                            None,
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Basic(BasicType::Int),
                                    TypeQualifiers::empty()
                                ),
                                vec![Deriv::Func(FuncDeclParams::Ansi {
                                    params: vec![
                                        FuncParam(
                                            None,
                                            Some(DerivedType(
                                                QualifiedType(
                                                    UnqualifiedType::Custom("x".to_string()),
                                                    TypeQualifiers::empty()
                                                ),
                                                vec![]
                                            ))
                                        ),
                                        FuncParam(
                                            None,
                                            Some(DerivedType(
                                                QualifiedType(
                                                    UnqualifiedType::Basic(BasicType::Char),
                                                    TypeQualifiers::empty()
                                                ),
                                                vec![]
                                            ))
                                        )
                                    ],
                                    is_variadic: false
                                })]
                            ))
                        )],
                        is_variadic: false
                    })],
                    None
                )]
            ))
        ]
    );
}

#[test]
fn test_function_definition() {
    assert_eq!(
        parse_external_declarations(r#"foo() {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![]
            ),
            FuncDefParams::Unspecified
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo() {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![]
            ),
            FuncDefParams::Unspecified
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(int a, char b) {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![]
            ),
            FuncDefParams::Ansi {
                params: vec![
                    FuncParam(
                        Some("a".to_string()),
                        Some(DerivedType(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Int),
                                TypeQualifiers::empty()
                            ),
                            vec![]
                        ))
                    ),
                    FuncParam(
                        Some("b".to_string()),
                        Some(DerivedType(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Char),
                                TypeQualifiers::empty()
                            ),
                            vec![]
                        ))
                    )
                ],
                is_variadic: false
            }
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(a) {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![]
            ),
            FuncDefParams::KnR {
                param_names: vec!["a".to_string()],
                decls: vec![],
            }
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(a, b) int a {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![]
            ),
            FuncDefParams::KnR {
                param_names: vec!["a".to_string(), "b".to_string()],
                decls: vec![KnRFuncParamDecl(
                    QualifiedType(
                        UnqualifiedType::Basic(BasicType::Int),
                        TypeQualifiers::empty()
                    ),
                    vec![("a".to_string(), vec![])]
                )],
            }
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void foo(a, b, c) short c; char *const a, b; {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![]
            ),
            FuncDefParams::KnR {
                param_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                decls: vec![
                    KnRFuncParamDecl(
                        QualifiedType(
                            UnqualifiedType::Basic(BasicType::Short),
                            TypeQualifiers::empty()
                        ),
                        vec![("c".to_string(), vec![])]
                    ),
                    KnRFuncParamDecl(
                        QualifiedType(
                            UnqualifiedType::Basic(BasicType::Char),
                            TypeQualifiers::empty()
                        ),
                        vec![
                            ("a".to_string(), vec![Deriv::Ptr(TypeQualifiers::CONST)]),
                            ("b".to_string(), vec![])
                        ]
                    )
                ],
            }
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"void *foo() {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Void),
                    TypeQualifiers::empty()
                ),
                vec![Deriv::Ptr(TypeQualifiers::empty())]
            ),
            FuncDefParams::Unspecified
        ))]
    );

    assert_eq!(
        parse_external_declarations(r#"short (*foo())() {}"#),
        vec![ExtDecl::FuncDef(FuncDef(
            "foo".to_string(),
            None,
            FuncSpecifiers::empty(),
            DerivedType(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Short),
                    TypeQualifiers::empty()
                ),
                vec![
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty())
                ]
            ),
            FuncDefParams::Unspecified
        ))]
    );
}

#[test]
fn test_function_pointer_declaration() {
    assert_eq!(
        parse_external_declarations(r#"int (*foo)();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty())
                ],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"int (*(foo))();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty())
                ],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"int (*(*bar)())();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "bar".to_string(),
                vec![
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty())
                ],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"int (*foo())();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Func(FuncDeclParams::Unspecified)
                ],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"char const * (**hogehoge)();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Char),
                TypeQualifiers::CONST
            ),
            vec![(
                "hogehoge".to_string(),
                vec![
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Ptr(TypeQualifiers::empty())
                ],
                None
            )]
        ))]
    );
}

#[test]
fn test_array_declaration() {
    assert_eq!(
        parse_external_declarations(r#"int foo[10];"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![Deriv::Array(ArraySize::Fixed(10))],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"char foo[1][3];"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Char),
                TypeQualifiers::empty()
            ),
            vec![(
                "foo".to_string(),
                vec![
                    Deriv::Array(ArraySize::Fixed(3)),
                    Deriv::Array(ArraySize::Fixed(1))
                ],
                None
            )]
        ))]
    );

    assert_eq!(
        // Unspecified size arrays should only in function parameters
        parse_external_declarations(r#"void bar(short foo[]);"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Void),
                TypeQualifiers::empty()
            ),
            vec![(
                "bar".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![FuncParam(
                        Some("foo".to_string()),
                        Some(DerivedType(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Short),
                                TypeQualifiers::empty()
                            ),
                            vec![Deriv::Array(ArraySize::Unspecified)]
                        ))
                    )],
                    is_variadic: false
                })],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"char *(*abcdef[3])();"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Char),
                TypeQualifiers::empty()
            ),
            vec![(
                "abcdef".to_string(),
                vec![
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Func(FuncDeclParams::Unspecified),
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Array(ArraySize::Fixed(3))
                ],
                None
            )]
        ))]
    );
}

#[test]
fn test_simple_type_definition() {
    assert_eq!(
        parse_external_declarations(r#"typedef signed *truc();"#),
        vec![ExtDecl::TypeDef(TypeDecl(
            QualifiedType(
                UnqualifiedType::Basic(BasicType::SignedInt),
                TypeQualifiers::empty()
            ),
            vec![(
                "truc".to_string(),
                vec![
                    Deriv::Ptr(TypeQualifiers::empty()),
                    Deriv::Func(FuncDeclParams::Unspecified)
                ]
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"typedef int *ptr; const ptr foo;"#),
        vec![
            ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![("ptr".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])]
            )),
            ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Custom("ptr".to_string()),
                    TypeQualifiers::CONST
                ),
                vec![("foo".to_string(), vec![], None)]
            ))
        ]
    );
    assert_eq!(
        parse_external_declarations(r#"typedef int i, *ptr; ptr foo, *bar;"#),
        vec![
            ExtDecl::TypeDef(TypeDecl(
                QualifiedType(
                    UnqualifiedType::Basic(BasicType::Int),
                    TypeQualifiers::empty()
                ),
                vec![
                    ("i".to_string(), vec![]),
                    ("ptr".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())])
                ]
            )),
            ExtDecl::Decl(Decl(
                None,
                QualifiedType(
                    UnqualifiedType::Custom("ptr".to_string()),
                    TypeQualifiers::empty()
                ),
                vec![
                    ("foo".to_string(), vec![], None),
                    (
                        "bar".to_string(),
                        vec![Deriv::Ptr(TypeQualifiers::empty())],
                        None
                    )
                ]
            ))
        ]
    );
    assert_eq!(
        // A typed declared and used in one typedef
        parse_external_declarations(r#"typedef int *foo, bar(foo x);"#),
        vec![ExtDecl::TypeDef(TypeDecl(
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![
                ("foo".to_string(), vec![Deriv::Ptr(TypeQualifiers::empty())]),
                (
                    "bar".to_string(),
                    vec![Deriv::Func(FuncDeclParams::Ansi {
                        params: vec![FuncParam(
                            Some("x".to_string()),
                            Some(DerivedType(
                                QualifiedType(
                                    UnqualifiedType::Custom("foo".to_string()),
                                    TypeQualifiers::empty()
                                ),
                                vec![]
                            ))
                        )],
                        is_variadic: false
                    })]
                )
            ]
        ))]
    );
    assert_eq!(
        // typedef, basic types keywords can be in any order 😢
        parse_external_declarations(r#"long typedef long unsigned foo;"#),
        vec![ExtDecl::TypeDef(TypeDecl(
            QualifiedType(
                UnqualifiedType::Basic(BasicType::UnsignedLongLong),
                TypeQualifiers::empty()
            ),
            vec![("foo".to_string(), vec![])]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"typedef long int __darwin_ptrdiff_t;"#),
        vec![ExtDecl::TypeDef(TypeDecl(
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Long),
                TypeQualifiers::empty()
            ),
            vec![("__darwin_ptrdiff_t".to_string(), vec![])]
        ))]
    );
}

#[test]
fn test_tag_definition() {
    assert_eq!(
        parse_external_declarations(r#"enum foo;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(Some("foo".to_string()), None)),
                TypeQualifiers::empty()
            ),
            vec![]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"enum foo bar;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(Some("foo".to_string()), None)),
                TypeQualifiers::empty()
            ),
            vec![("bar".to_string(), vec![], None)]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"enum foo { a, b = 10, c } bar;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(
                    Some("foo".to_string()),
                    Some(vec![
                        EnumItem("a".to_string(), None),
                        EnumItem("b".to_string(), Some(10)),
                        EnumItem("c".to_string(), None),
                    ])
                )),
                TypeQualifiers::empty()
            ),
            vec![("bar".to_string(), vec![], None)]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"enum foo { a, b = 10, c } bar(void);"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(
                    Some("foo".to_string()),
                    Some(vec![
                        EnumItem("a".to_string(), None),
                        EnumItem("b".to_string(), Some(10)),
                        EnumItem("c".to_string(), None),
                    ])
                )),
                TypeQualifiers::empty()
            ),
            vec![(
                "bar".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![],
                    is_variadic: false
                })],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"enum { a, b = 10, c } bar(void);"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(
                    None,
                    Some(vec![
                        EnumItem("a".to_string(), None),
                        EnumItem("b".to_string(), Some(10)),
                        EnumItem("c".to_string(), None),
                    ])
                )),
                TypeQualifiers::empty()
            ),
            vec![(
                "bar".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![],
                    is_variadic: false
                })],
                None
            )]
        ))]
    );
    assert_eq!(
        // enum in function parameters - note that in that case the enum is only usable inside the function.
        parse_external_declarations(r#"enum foo { a, b = 10, c } bar(enum hoge { x });"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(
                    Some("foo".to_string()),
                    Some(vec![
                        EnumItem("a".to_string(), None),
                        EnumItem("b".to_string(), Some(10)),
                        EnumItem("c".to_string(), None),
                    ])
                )),
                TypeQualifiers::empty()
            ),
            vec![(
                "bar".to_string(),
                vec![Deriv::Func(FuncDeclParams::Ansi {
                    params: vec![FuncParam(
                        None,
                        Some(DerivedType(
                            QualifiedType(
                                UnqualifiedType::Tag(Tag::Enum(
                                    Some("hoge".to_string()),
                                    Some(vec![EnumItem("x".to_string(), None)])
                                )),
                                TypeQualifiers::empty()
                            ),
                            vec![]
                        ))
                    )],
                    is_variadic: false
                })],
                None
            )]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"enum foo { a, b = 10, c };"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Enum(
                    Some("foo".to_string()),
                    Some(vec![
                        EnumItem("a".to_string(), None),
                        EnumItem("b".to_string(), Some(10)),
                        EnumItem("c".to_string(), None),
                    ])
                )),
                TypeQualifiers::empty()
            ),
            vec![]
        ))]
    );
    assert_eq!(
        parse_external_declarations(r#"struct foo { int a : 1, b; };"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Struct(
                    Some("foo".to_string()),
                    Some(vec![TagItemDecl(
                        QualifiedType(
                            UnqualifiedType::Basic(BasicType::Int),
                            TypeQualifiers::empty()
                        ),
                        vec![
                            ("a".to_string(), vec![], Some(1)),
                            ("b".to_string(), vec![], None),
                        ]
                    )])
                )),
                TypeQualifiers::empty()
            ),
            vec![]
        ))]
    );
    // self-referencing struct
    assert_eq!(
        parse_external_declarations(r#"struct s { struct s *next };"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Struct(
                    Some("s".to_string()),
                    Some(vec![TagItemDecl(
                        QualifiedType(
                            UnqualifiedType::Tag(Tag::Struct(Some("s".to_string()), None)),
                            TypeQualifiers::empty()
                        ),
                        vec![(
                            "next".to_string(),
                            vec![Deriv::Ptr(TypeQualifiers::empty())],
                            None
                        )]
                    )])
                )),
                TypeQualifiers::empty()
            ),
            vec![]
        ))]
    );
    assert_eq!(
        parse_external_declarations(
            r#"union { int x : 2; struct { int a, *b }; char * const z } foo;"#
        ),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Tag(Tag::Union(
                    None,
                    Some(vec![
                        TagItemDecl(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Int),
                                TypeQualifiers::empty()
                            ),
                            vec![("x".to_string(), vec![], Some(2)),]
                        ),
                        TagItemDecl(
                            QualifiedType(
                                UnqualifiedType::Tag(Tag::Struct(
                                    None,
                                    Some(vec![TagItemDecl(
                                        QualifiedType(
                                            UnqualifiedType::Basic(BasicType::Int),
                                            TypeQualifiers::empty()
                                        ),
                                        vec![
                                            ("a".to_string(), vec![], None),
                                            (
                                                "b".to_string(),
                                                vec![Deriv::Ptr(TypeQualifiers::empty())],
                                                None
                                            )
                                        ]
                                    )])
                                )),
                                TypeQualifiers::empty()
                            ),
                            vec![]
                        ),
                        TagItemDecl(
                            QualifiedType(
                                UnqualifiedType::Basic(BasicType::Char),
                                TypeQualifiers::empty()
                            ),
                            vec![(
                                "z".to_string(),
                                vec![Deriv::Ptr(TypeQualifiers::CONST)],
                                None
                            )]
                        )
                    ])
                )),
                TypeQualifiers::empty()
            ),
            vec![("foo".to_string(), vec![], None)]
        ))]
    );
}

#[test]
fn test_variable_initialization() {
    assert_eq!(
        parse_external_declarations(r#"int abcd = 42;"#),
        vec![ExtDecl::Decl(Decl(
            None,
            QualifiedType(
                UnqualifiedType::Basic(BasicType::Int),
                TypeQualifiers::empty()
            ),
            vec![("abcd".to_string(), vec![], Some(42))],
        ))]
    );
}
