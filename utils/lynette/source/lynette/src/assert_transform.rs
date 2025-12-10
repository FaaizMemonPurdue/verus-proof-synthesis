use std::path::PathBuf;

use crate::deghost::{deghost_merge_files, remove_verus_macro};
use crate::utils::*;
use proc_macro2::{Span, TokenStream};
use quote::format_ident;
use syn_verus::spanned::Spanned;
use syn_verus::visit_mut::{self, VisitMut};
use syn_verus::{
    Attribute, BinOp, Block, Expr, ExprBinary, ExprMacro, FnArg, FnArgKind, FnMode, Ident, Item, ItemFn, Pat, PatIdent, PatType, ReturnType, Specification, Stmt, Type, TypeArray, TypeTuple, UnOp, UseTree, parse_quote
};
use serde::{Deserialize, Serialize};

fn mk_assert_semi(expr: &Expr, message: Option<&str>) -> Stmt {
    let message = message.unwrap_or("");
    let mut assert_mac: ExprMacro = parse_quote! {
        assert!(#expr)
    };
    let attr: Attribute = parse_quote! {
        #[a(#message)]
    };
    assert_mac.attrs = vec![attr];
    let assert_expr = Expr::Macro(assert_mac);
    Stmt::Semi(assert_expr, Default::default())
}

fn and_expr(lhs: &Expr, rhs: &Expr) -> Expr {
    let result = parse_quote! {
        (#lhs) && (#rhs)
    };
    Expr::Binary(result)
}

fn imply_expr(lhs: &Expr, rhs: &Expr) -> Expr {
    parse_quote! {
        !(#lhs) || (#rhs)
    }
}

fn put_parens(expr: &Expr) -> Expr {
    parse_quote! {
        (#expr)
    }
}

fn true_expr() -> Expr {
    parse_quote! {
        true
    }
}

fn add_cast(expr: &Expr, typ: &str) -> Expr {
    let typ = format_ident!("{}", typ);
    parse_quote! {
        (#expr as #typ)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Loc {
    line: usize,
    col: usize,
}

impl From<Span> for Loc {
    fn from(value: Span) -> Self {
        Loc {
            line: value.start().line,
            col: value.start().column,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClauseInfo {
    clause: String,
    assert_loc: Loc,
    clause_loc: Loc,
}

#[derive(Debug, Clone, Default)]
struct Clauses {
    metadata: Vec<ClauseInfo>,
    clauses: Vec<Expr>,
}

impl Clauses {
    fn from_spec(spec: &Specification, context: &TransformContext) -> Self {
        Clauses {
            metadata: spec.exprs
                .iter()
                .map(|expr| ClauseInfo {
                    clause: expr.span().source_text().unwrap_or(String::new()),
                    // just default for now, figured out later
                    assert_loc: Loc::default(),
                    clause_loc: Loc::from(expr.span()),
                })
                .collect(),
            clauses: spec.exprs
                .iter()
                .map(|expr| {
                    let mut expr = expr.clone();
                    update_expression_spec(&mut expr, context);
                    expr
                })
                .collect(),
        }
    }

    fn asserts(&self) -> impl Iterator<Item = Stmt> {
        self.clauses.iter()
            .zip(self.metadata.iter())
            .map(|(expr, metadata)| {
                let metadata_string = serde_json::to_string(metadata).unwrap();
                mk_assert_semi(expr, Some(&metadata_string))
            })
    }

    fn insert_in_block(&self, block: &mut Block, index: usize) {
        block.stmts.splice(index..index, self.asserts());
    }

    fn as_single_expr(&self) -> Expr {
        self.clauses.iter()
            .fold(true_expr(), |accumulator, elem| and_expr(&accumulator, elem))
    }
}

struct CastVisitor;

// FIXME: this whole thing is hacky fix to get types to check out cause we don't have much time
impl VisitMut for CastVisitor {
    fn visit_expr_mut(&mut self, expr: &mut syn_verus::Expr) {
        visit_mut::visit_expr_mut(self, expr);

        match expr {
            Expr::Index(expr_index) => {
                expr_index.index = Box::new(add_cast(&expr_index.index, "usize"));
            }
            Expr::MethodCall(expr_method_call) => {
                if expr_method_call.method.to_string() == "len" {
                    *expr = add_cast(expr, "i128");
                }
            }
            Expr::Path(_expr_path) => {
                // FIXME: figure out this casting issue
                // *expr = add_cast(expr, "i128");
            }
            _ => (),
        }
    }
}

fn transform_quantifier(clause: Expr, is_forall: bool, quantifier_iterations: u32) -> Expr {
    let Expr::Closure(mut closure) = clause else {
        panic!("quantifier must be followed by closure");
    };

    let closure_ident = format_ident!("condition");
    let result_ident = format_ident!("result");
    let arg_idents = (0..closure.inputs.len()).map(|n| format_ident!("arg{n}"));

    let exit_condition = !is_forall;
    let start_value = is_forall;
    let update_value = !start_value;

    let args = arg_idents.clone();
    let mut loop_body: Expr = parse_quote! {{
        if #closure_ident( #(#args),* ) == #exit_condition {
            #result_ident = #update_value;
            break;
        }
    }};

    for (arg, arg_name) in closure.inputs.iter_mut().zip(arg_idents) {
        // TODO: based on arg pick range
        loop_body = parse_quote! {{
            for #arg_name in 0..(#quantifier_iterations as i128) {
                for #arg_name in [#arg_name, -#arg_name] {
                    #loop_body
                }
            }
        }};

        if let Pat::Type(pat_type) = arg {
            *arg = *pat_type.pat.clone();
        }
    }

    let result: Expr = parse_quote! {{
        let #closure_ident = #closure;
        let mut #result_ident = #start_value;
        #loop_body;
        #result_ident
    }};

    result
}

/// Attempts to transform verus specs into executable code which checks at runtime
struct SpecConversionVisitor {
    quantifier_iterations: u32,
}

impl SpecConversionVisitor {
    fn new(context: &TransformContext) -> Self {
        SpecConversionVisitor { quantifier_iterations: context.quantifier_iterations }
    }
}

impl VisitMut for SpecConversionVisitor {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);
        match expr {
            Expr::Unary(expr_unary) => match expr_unary.op {
                UnOp::Forall(_) => {
                    *expr = transform_quantifier(
                        *expr_unary.expr.clone(),
                        true,
                        self.quantifier_iterations,
                    )
                }
                UnOp::Exists(_) => {
                    *expr = transform_quantifier(
                        *expr_unary.expr.clone(),
                        false,
                        self.quantifier_iterations,
                    )
                }
                _ => (),
            },
            Expr::Binary(expr_binary) => match expr_binary.op {
                BinOp::Imply(_) => *expr = imply_expr(&expr_binary.left, &expr_binary.right),
                _ => *expr = put_parens(&expr),
            },
            _ => (),
        }
    }
}

struct ComparisonConversionVisitor;

fn is_comparison(expr: &ExprBinary) -> bool {
    matches!(expr.op, BinOp::Lt(_) | BinOp::Gt(_) | BinOp::Le(_) | BinOp::Ge(_))
}

impl VisitMut for ComparisonConversionVisitor {
    fn visit_expr_binary_mut(&mut self, expr_binary: &mut ExprBinary) {
        if is_comparison(expr_binary) {
            let op = expr_binary.op;
            if let Expr::Binary(ref lhs) = *expr_binary.left
                && is_comparison(lhs)
            {
                let lhs_right = &*lhs.right;
                let rhs = &*expr_binary.right;
                *expr_binary = parse_quote! {
                    (#lhs) && (#lhs_right #op #rhs)
                };
            } else if let Expr::Binary(ref rhs) = *expr_binary.right
                && is_comparison(rhs)
            {
                let rhs_left = &*rhs.left;
                let lhs = &*expr_binary.left;
                *expr_binary = parse_quote! {
                    (#lhs #op #rhs_left) && (#rhs)
                }
            }
        }
        visit_mut::visit_expr_binary_mut(self, expr_binary);
    }
}

// converts verus specific things like quantifiers and such
fn update_expression_spec(expr: &mut Expr, context: &TransformContext) {
    ComparisonConversionVisitor.visit_expr_mut(expr);
    CastVisitor.visit_expr_mut(expr);
    SpecConversionVisitor::new(context).visit_expr_mut(expr);
}

fn make_crux_symbolic_argument_inner(arg: &Type, arg_name: Ident) -> Vec<Stmt> {
    let arg_name_str = arg_name.to_string();

    match arg {
        Type::Reference(ref_type) => {
            if let Type::Slice(slice_type) = &*ref_type.elem {
                let array_typ = Type::Array(TypeArray {
                    bracket_token: Default::default(),
                    elem: slice_type.elem.clone(),
                    semi_token: Default::default(),
                    len: parse_quote!(1),
                });

                let arr_name = format_ident!("symbolic_array_{arg_name}");

                parse_quote! {
                    let #arr_name = <#array_typ as Symbolic>::symbolic(#arg_name_str);
                    let #arg_name = crucible::symbolic::prefix(&#arr_name[..]);
                }
            } else {
                let ref_name = format_ident!("ref_{arg_name}");
                let mut inner_statements =
                    make_crux_symbolic_argument_inner(&ref_type.elem, ref_name.clone());
                inner_statements.push(parse_quote! {
                    let #arg_name = &#ref_name;
                });
                inner_statements
            }
        }
        typ => parse_quote! {
            let #arg_name = <#typ as Symbolic>::symbolic(#arg_name_str);
        },
    }
}

// FIXME: returning vec of stmt is a bit hacky
fn make_crux_symbolic_argument(arg: &FnArg, arg_name: Ident) -> Vec<Stmt> {
    let FnArgKind::Typed(arg) = &arg.kind else {
        panic!("recevier args not supported for testing in crux");
    };

    make_crux_symbolic_argument_inner(&arg.ty, arg_name)
}

/// Creates a crux test function which calls the given function with symbolic values
fn make_crux_test_fn(func: &ItemFn) -> ItemFn {
    let arg_names = (0..func.sig.inputs.len()).map(|i| format_ident!("arg{i}"));

    let arg_stmts = func
        .sig
        .inputs
        .iter()
        .zip(arg_names.clone())
        .map(|(arg, arg_name)| make_crux_symbolic_argument(arg, arg_name))
        .flatten();

    let name = func.sig.ident.clone();
    let test_name = format_ident!("test_{}", name);
    let mut result: ItemFn = parse_quote! {
        #[cfg_attr(crux, crux::test)]
        fn #test_name() {
            #name( #(#arg_names),* );
        }
    };

    result.block.stmts.splice(0..0, arg_stmts);
    result
}

fn make_wrapper_fn(
    func: &ItemFn,
    return_pattern: &Pat,
    return_type: &Type,
    requires_clauses: &Clauses,
    ensures_clauses: &Clauses,
    context: &TransformContext,
) -> Vec<ItemFn> {
    let mut wrapper_fn = func.clone();

    // change name
    let fn_name = func.sig.ident.clone();
    wrapper_fn.sig.ident = format_ident!("{}_assert_wrapper", fn_name);
    wrapper_fn.sig.output = ReturnType::Default;

    // change paramaters to just be a single ident, no patterns
    for (i, arg) in wrapper_fn.sig.inputs.iter_mut().enumerate() {
        match arg.kind {
            FnArgKind::Typed(ref mut arg) => {
                arg.pat = Box::new(Pat::Ident(PatIdent {
                    attrs: Vec::new(),
                    by_ref: None,
                    mutability: None,
                    ident: format_ident!("arg{i}"),
                    subpat: None,
                }))
            }
            FnArgKind::Receiver(_) => todo!(),
        }
    }

    let args = wrapper_fn.sig.inputs.iter().enumerate().map(|(i, arg)| match &arg.kind {
        FnArgKind::Typed(_) => format_ident!("arg{i}"),
        FnArgKind::Receiver(_) => format_ident!("self"),
    });

    let mut pre_check_fn = func.clone();
    let pre_check_ident = format_ident!("{}_assert_pre_check", fn_name);
    pre_check_fn.sig.ident = pre_check_ident.clone();
    let bool_type: Type = parse_quote! { bool };
    pre_check_fn.sig.output = ReturnType::Type(Default::default(), None, None, Box::new(bool_type));
    pre_check_fn.block.stmts = vec![Stmt::Expr(requires_clauses.as_single_expr())];

    let mut post_check_fn = func.clone();
    let post_check_ident = format_ident!("{}_assert_post_check", fn_name);
    post_check_fn.sig.ident = post_check_ident.clone();
    post_check_fn.sig.output = ReturnType::Default;
    post_check_fn.sig.inputs.push(FnArg {
        tracked: None,
        kind: FnArgKind::Typed(PatType {
            pat: Box::new(return_pattern.clone()),
            ty: Box::new(return_type.clone()),
            attrs: Vec::new(),
            colon_token: Default::default(),
        }),
    });
    post_check_fn.block.stmts = ensures_clauses.asserts().collect::<Vec<_>>();

    let args2 = args.clone();
    let args3 = args.clone();
    let wrapper_body: Block = parse_quote! {
        {
            if #pre_check_ident( #(#args),* ) {
                let result = #fn_name( #(#args2),* );
                #post_check_ident( #(#args3),*, result );
            };
        }
    };
    // a bit hacky for some reason we can't put assert inside parse quote
    // wrapper_body.stmts.insert(1, ensures_stmt.clone());
    wrapper_fn.block = Box::new(wrapper_body);

    let mut out = vec![pre_check_fn, post_check_fn, wrapper_fn.clone()];
    if context.crux_test {
        out.push(make_crux_test_fn(&wrapper_fn));
    }
    out
}

fn reset_fn_sig(func: &mut ItemFn) -> (Pat, Box<Type>) {
    func.sig.erase_spec_fields();
    // erase_spec_fields doesn't set this back to default
    // ensures spec functions turn into regular functions
    // TODO: transform body of proof mode to executable code
    func.sig.mode = FnMode::Default;

    // get rid of named return types
    let (return_pattern, return_type) = match func.sig.output.clone() {
        // TODO: figure out what this tracked thing is
        ReturnType::Type(arrow, _, Some(pattern), return_type) => {
            let result = pattern.1;
            func.sig.output = ReturnType::Type(arrow, None, None, return_type.clone());
            (result, return_type)
        }
        _ => {
            let pattern = Pat::Ident(PatIdent {
                attrs: Vec::new(),
                by_ref: None,
                mutability: None,
                ident: format_ident!("result"),
                subpat: None,
            });
            let return_type = Type::Tuple(TypeTuple {
                paren_token: Default::default(),
                elems: Default::default(),
            });
            (pattern, Box::new(return_type))
        }
    };

    (return_pattern, return_type)
}

fn transform_fn(func: &mut ItemFn, context: &TransformContext) -> Vec<ItemFn> {
    // for spec function just translate body and do nothing else
    if matches!(func.sig.mode, FnMode::Spec(_)) {
        ComparisonConversionVisitor.visit_block_mut(&mut func.block);
        CastVisitor.visit_block_mut(&mut func.block);
        SpecConversionVisitor::new(context).visit_block_mut(&mut func.block);
        reset_fn_sig(func);

        return Vec::new();
    }

    let requires_clauses = match &func.sig.requires {
        Some(spec) => Clauses::from_spec(&spec.exprs, context),
        None => Clauses::default(),
    };

    let ensures_clauses = match &func.sig.ensures {
        Some(spec) => Clauses::from_spec(&spec.exprs, context),
        None => Clauses::default(),
    };

    // get rid of named return types
    let (return_pattern, return_type) = reset_fn_sig(func);

    InvariantTransformer { context }.visit_block_mut(&mut func.block);

    make_wrapper_fn(func, &return_pattern, &return_type, &requires_clauses, &ensures_clauses, context)
}

/// Transforms loop invariants into asserts
struct InvariantTransformer<'a> {
    context: &'a TransformContext,
}

impl VisitMut for InvariantTransformer<'_> {
    fn visit_expr_while_mut(&mut self, expr_while: &mut syn_verus::ExprWhile) {
        if let Some(invariants) = &expr_while.invariant {
            let invariant_clauses = Clauses::from_spec(&invariants.exprs, self.context);

            invariant_clauses.insert_in_block(&mut expr_while.body, 0);
            let body_len = expr_while.body.stmts.len();
            invariant_clauses.insert_in_block(&mut expr_while.body, body_len);
        }
        expr_while.invariant = None;

        visit_mut::visit_expr_while_mut(self, expr_while);
    }
}

fn is_vstd_import(item: &Item) -> bool {
    let Item::Use(item_use) = item else {
        return false;
    };

    let UseTree::Path(use_path) = &item_use.tree else {
        return false;
    };

    use_path.ident.to_string() == "vstd"
}

fn assert_transform_parsed_file(file: &mut syn_verus::File, context: &TransformContext) {
    let mut new_functions = Vec::new();
    for item in &mut file.items {
        if let Item::Fn(func) = item {
            new_functions
                .extend(transform_fn(func, context).into_iter().map(|function| Item::Fn(function)));
        }
    }
    file.items.extend(new_functions);
}

struct AssertLocation {
    span: Span,
    metadata: Option<ClauseInfo>,
}

// extracts and metadata for each assert in the file, removes it if present
// used to match metadata to real assert
#[derive(Default)]
struct AssertPostprocessVisitor {
    assertions: Vec<AssertLocation>,
}

impl syn::visit_mut::VisitMut for AssertPostprocessVisitor {
    fn visit_expr_macro_mut(&mut self, expr: &mut syn::ExprMacro) {
        // FIXME: actually inspect attr path, so we get the right one, too lazy rn
        let metadata = if let Some(attr) = expr.attrs.pop() {
            let metadata_str = attr.tokens.to_string().replace("\\\"", "\"");
            println!("{}", metadata_str);
            let metadata = serde_json::from_str(&metadata_str[2..metadata_str.len() - 2]).unwrap();
            Some(metadata)
        } else {
            None
        };

        self.assertions.push(AssertLocation {
            span: expr.span(),
            metadata,
        });

        syn::visit_mut::visit_expr_macro_mut(self, expr);
    }
}

#[derive(Debug, Serialize)]
struct TransformedMetadata {
    clause_assertions: Vec<ClauseInfo>,
}

fn postprocess_file(file_data: &str) -> Result<(String, TransformedMetadata), Error> {
    use syn::visit_mut::VisitMut;
    use quote::ToTokens;

    let mut parsed_file_with_metadata = load_file_str_rust(file_data.to_string());

    let mut metadata_visitor = AssertPostprocessVisitor::default();
    let mut no_metadata_visitor = AssertPostprocessVisitor::default();

    metadata_visitor.visit_file_mut(&mut parsed_file_with_metadata);

    let mut ts = TokenStream::new();
    parsed_file_with_metadata.to_tokens(&mut ts);
    let formatted_no_metadata = format_token_stream(&ts, Formatter::RustFmt);

    let mut parsed_file_no_metadata = load_file_str_rust(formatted_no_metadata.clone());
    no_metadata_visitor.visit_file_mut(&mut parsed_file_no_metadata);

    assert!(metadata_visitor.assertions.len() == no_metadata_visitor.assertions.len());

    let clause_assertions = metadata_visitor.assertions.iter()
        .zip(no_metadata_visitor.assertions.iter())
        .filter_map(|(old_location, new_location)| match &old_location.metadata {
            Some(metadata) => Some(ClauseInfo {
                clause: metadata.clause.clone(),
                clause_loc: metadata.clause_loc,
                assert_loc: Loc::from(new_location.span),
            }),
            None => None,
        })
        .collect::<Vec<_>>();

    let metadata = TransformedMetadata {
        clause_assertions,
    };

    Ok((formatted_no_metadata, metadata))
}

// need &PathBuf for helper functions
// original helpers are written weirdly
fn assert_transform_file(
    old_file_path: &PathBuf,
    new_file_path: &PathBuf,
    metadata_path: &PathBuf,
    context: &mut TransformContext,
) -> Result<(), Error> {
    let mut parsed_file = fload_file(old_file_path)?;
    // remove vstd imports
    parsed_file.items.retain(|item| !is_vstd_import(item));

    if context.crux_test {
        // add crux imports
        parsed_file.items.insert(
            0,
            parse_quote! {
                #[cfg(crux)] extern crate crucible;
            },
        );
        parsed_file.items.insert(
            1,
            parse_quote! {
                #[cfg(crux)] use crucible::*;
            },
        );
    }

    let pure_file = remove_verus_macro(&parsed_file);

    let mut parsed_verus_blocks = extract_verus_macro(&parsed_file)?;
    for file in parsed_verus_blocks.iter_mut() {
        assert_transform_parsed_file(file, context);
    }

    let new_file = deghost_merge_files(&pure_file, parsed_verus_blocks);
    let new_code = fprint_file(&new_file, Formatter::RustFmt);

    let (postprocessed_code, metadata) = postprocess_file(&new_code)?;
    let metadata = serde_json::to_string(&metadata).unwrap();

    std::fs::write(new_file_path, postprocessed_code)?;
    std::fs::write(metadata_path, metadata)?;

    Ok(())
}

pub struct TransformContext {
    /// Generate code to use crux to find counterexamples
    pub crux_test: bool,
    /// Number of quantifier iterations to try
    pub quantifier_iterations: u32,
}

pub fn do_assert_transform_file(
    old_file_path: &PathBuf,
    new_file_path: &PathBuf,
    metadata_path: &PathBuf,
    context: &mut TransformContext,
) {
    if let Err(error) = assert_transform_file(old_file_path, new_file_path, metadata_path, context) {
        panic!("error transforming file: {}", error);
    }
}
