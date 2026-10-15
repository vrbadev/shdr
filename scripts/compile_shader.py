#!/usr/bin/env python3
"""Compile native GLSL compute sources into C++11 adapters and embedded GPU programs.

The frontend translates native layout, buffer, uniform, main and swizzle syntax
for the documented independent-invocation subset. Interface validation precedes
C++ compilation. Applications execute the embedded CPU and GPU programs.
"""
import argparse # Read the build or test command options.
import ast # Evaluate restricted metadata expressions through their syntax tree.
import hashlib # Give generated names stable collision resistant suffixes.
import operator # Apply only the arithmetic operations allowed in metadata.
from pathlib import Path # Read shader inputs and write generated adapter paths.
import re # Recognize native declarations while retaining their source offsets.


class ShaderError(ValueError): # Group these parser operations or diagnostic values.
    """A native shader declaration outside the supported portable contract."""


def mask_noncode(source): # Mask comments and text while preserving source offsets.
    """Preserve offsets/newlines while hiding comments and quoted literals."""
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'' # Compute pattern for the following operation.
    return re.sub(pattern, lambda m: re.sub(r'[^\n]', ' ', m.group()), source) # Return this computed value to the caller.


def layout_values(text): # Parse the comma separated GLSL layout qualifiers.
    result = {} # Compute result for the following operation.
    for item in text.split(','): # Process each item in this sequence.
        if not item.strip(): # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        key, _, value = item.strip().partition('=') # Apply strip to these arguments.
        key = key.strip() # Compute key for the following operation.
        if key in result: # Select this branch when its condition holds.
            raise ShaderError('duplicate layout qualifier: ' + key) # Report the shader declaration validation error.
        result[key.strip()] = value.strip() # Apply strip to these arguments.
    return result # Return this computed value to the caller.


def integer(text, label, default=None, constants=None): # Evaluate the limited integer expressions accepted by resource metadata.
    if text is None and default is not None: # Select this branch when its condition holds.
        return default # Return this computed value to the caller.
    constants = constants or {} # Compute constants for the following operation.
    operations = { # Compute operations for the following operation.
        ast.Add: operator.add, # Supply ast for the continued expression.
        ast.Sub: operator.sub, # Supply ast for the continued expression.
        ast.Mult: operator.mul, # Supply ast for the continued expression.
        ast.Div: operator.floordiv, # Supply ast for the continued expression.
        ast.FloorDiv: operator.floordiv, # Supply ast for the continued expression.
        ast.Mod: operator.mod, # Supply ast for the continued expression.
        ast.LShift: operator.lshift, # Supply ast for the continued expression.
        ast.RShift: operator.rshift, # Supply ast for the continued expression.
        ast.BitOr: operator.or_, # Supply ast for the continued expression.
        ast.BitAnd: operator.and_, # Supply ast for the continued expression.
        ast.BitXor: operator.xor, # Supply ast for the continued expression.
    } # Finish this expression or collection.

    def evaluate(expression, seen=()): # Resolve constants recursively through the restricted expression tree.
        expression = re.sub(r'\b(0[xX][0-9a-fA-F]+|[0-9]+)[uU]\b', r'\1', expression) # Compute expression for the following operation.

        def visit(node): # Evaluate one permitted expression node.
            if isinstance(node, ast.Constant) and isinstance(node.value, int): # Select this branch when its condition holds.
                return node.value # Return this computed value to the caller.
            if ( # Select this branch when its condition holds.
                isinstance(node, ast.Name) # Apply isinstance to these arguments.
                and node.id in constants # Supply and for the continued expression.
                and node.id not in seen # Supply and for the continued expression.
            ): # Finish this expression or collection.
                return evaluate(constants[node.id], seen + (node.id,)) # Return this computed value to the caller.
            if isinstance(node, ast.BinOp) and type(node.op) in operations: # Select this branch when its condition holds.
                left, right = visit(node.left), visit(node.right) # Apply visit to these arguments.
                if ( # Select this branch when its condition holds.
                    isinstance(node.op, (ast.LShift, ast.RShift)) # Apply isinstance to these arguments.
                    and not 0 <= right < 32 # Supply and for the continued expression.
                ): # Finish this expression or collection.
                    raise ValueError() # Report the shader declaration validation error.
                return operations[type(node.op)](left, right) # Return this computed value to the caller.
            if isinstance(node, ast.UnaryOp) and isinstance( # Select this branch when its condition holds.
                node.op, (ast.UAdd, ast.USub, ast.Invert) # Supply node for the continued expression.
            ): # Finish this expression or collection.
                value = visit(node.operand) # Compute value for the following operation.
                return ( # Return this computed value to the caller.
                    value # Supply value for the continued expression.
                    if isinstance(node.op, ast.UAdd) # Select this branch when its condition holds.
                    else -value if isinstance(node.op, ast.USub) else ~value # Apply isinstance to these arguments.
                ) # Finish this expression or collection.
            raise ValueError() # Report the shader declaration validation error.

        return visit(ast.parse(expression, mode='eval').body) # Return this computed value to the caller.

    try: # Attempt the operation with explicit error handling below.
        value = evaluate(text or '') # Compute value for the following operation.
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, RecursionError): # Convert this failure into the expected diagnostic.
        raise ShaderError( # Report the shader declaration validation error.
            label # Supply label for the continued expression.
            + ' must be a literal nonnegative integer or an integer constant expression'
        ) # Finish this expression or collection.
    if value < 0: # Select this branch when its condition holds.
        raise ShaderError(label + ' must be a nonnegative integer') # Report the shader declaration validation error.
    if value > 0xFFFFFFFF: # Select this branch when its condition holds.
        raise ShaderError(label + ' exceeds GLSL uint') # Report the shader declaration validation error.
    return value # Return this computed value to the caller.


STORAGE = {'float', 'int', 'uint', 'vec2', 'vec4', 'ivec2', 'ivec4', 'uvec2', 'uvec4'} # Compute STORAGE for the following operation.
UNIFORMS = ( # Compute UNIFORMS for the following operation.
    STORAGE # Supply STORAGE for the continued expression.
    | {'bool', 'vec3', 'ivec3', 'uvec3', 'bvec2', 'bvec3', 'bvec4'} # Supply bool for the continued expression.
    | { # Supply this value for the continued expression.
        'mat2',
        'mat3',
        'mat4',
        *(f'mat{c}x{r}' for c in range(2, 5) for r in range(2, 5)),
    } # Finish this expression or collection.
) # Finish this expression or collection.
IMAGES = { # Compute IMAGES for the following operation.
    'r32f': ('float', 1, 'image2D'), # Supply r32f for the continued expression.
    'r32i': ('int', 1, 'iimage2D'), # Supply r32i for the continued expression.
    'r32ui': ('unsigned', 1, 'uimage2D'), # Supply r32ui for the continued expression.
    'rgba32f': ('float', 4, 'image2D'), # Supply rgba32f for the continued expression.
    'rgba32i': ('int', 4, 'iimage2D'), # Supply rgba32i for the continued expression.
    'rgba32ui': ('unsigned', 4, 'uimage2D'), # Supply rgba32ui for the continued expression.
} # Finish this expression or collection.
QUALIFIERS = ( # Compute QUALIFIERS for the following operation.
    r'(?:(?:readonly|writeonly|coherent|volatile|restrict|highp|mediump|lowp)\s+)*'
) # Finish this expression or collection.


def access_mode(qualifiers): # Translate resource access qualifiers into the runtime access mode.
    words = qualifiers.split() # Compute words for the following operation.
    if 'readonly' in words and 'writeonly' in words: # Select this branch when its condition holds.
        raise ShaderError('resource cannot be both readonly and writeonly') # Report the shader declaration validation error.
    return ( # Return this computed value to the caller.
        'readonly'
        if 'readonly' in words # Select this branch when its condition holds.
        else 'writeonly' if 'writeonly' in words else 'readwrite' # Supply else for the continued expression.
    ) # Finish this expression or collection.


def member_receiver(source, position): # Find the object receiving a member or swizzle operation.
    """Return a bare or parenthesized identifier immediately before a member dot."""
    prefix = source[:position].rstrip() # Compute prefix for the following operation.
    parentheses = 0 # Compute parentheses for the following operation.
    while prefix.endswith(')'): # Continue while the parser still has matching input.
        parentheses += 1 # Compute parentheses for the following operation.
        prefix = prefix[:-1].rstrip() # Compute prefix for the following operation.
    identifier = re.search(r'\b([A-Za-z_]\w*)$', prefix) # Compute identifier for the following operation.
    if not identifier: # Select this branch when its condition holds.
        return None # Return this computed value to the caller.
    before = prefix[: identifier.start()] # Compute before for the following operation.
    for _ in range(parentheses): # Process each   in this sequence.
        before = before.rstrip() # Compute before for the following operation.
        if not before.endswith('('): # Select this branch when its condition holds.
            return None # Return this computed value to the caller.
        before = before[:-1] # Compute before for the following operation.
    # Keep this member access attached to its enclosing object.
    if before.rstrip().endswith('.'): # Select this branch when its condition holds.
        return None # Return this computed value to the caller.
    return identifier.group(1) # Return this computed value to the caller.


def primary_expression_start(code, end): # Locate the complete receiver of a native array length query.
    """Find the postfix receiver within its surrounding expression boundaries."""
    while end and code[end - 1].isspace(): # Ignore whitespace between a receiver and its member dot.
        end -= 1 # Move toward the last token belonging to the receiver.
    if not end: # A member query must have an expression before its dot.
        raise ShaderError('array length query has no receiver') # Diagnose incomplete native syntax before emitting C++.
    if code[end - 1] in ')]': # Receiver expressions can end in a call, grouping or array index.
        closing = code[end - 1] # Remember which delimiter closes this receiver expression.
        opening = '(' if closing == ')' else '[' # Select the matching opening delimiter.
        depth, start = 1, end - 1 # Begin the balanced scan before the closing delimiter.
        while start and depth: # Walk backward until this exact delimiter pair is balanced.
            start -= 1 # Inspect the preceding character in the original masked source.
            if code[start] == closing: # Count nested closing delimiters of the same kind.
                depth += 1 # Keep the outer delimiter open until its own match is reached.
            elif code[start] == opening: # Match one nested opening delimiter.
                depth -= 1 # Remove this balanced level from the search.
        if depth: # A missing opener would make the native expression incomplete.
            raise ShaderError('unbalanced array length receiver') # Stop at the existing expression boundary.
        prefix = code[:start].rstrip() # Inspect the expression immediately preceding the opening delimiter.
        if closing == ']': # Indexing belongs to its complete array or vector receiver.
            start = primary_expression_start(code, start) # Include earlier member access, calls and indices.
        elif prefix: # Parentheses may group an expression or contain a function call's arguments.
            identifier = re.search(r'\b([A-Za-z_]\w*)$', prefix) # Detect a callee while distinguishing statement keywords.
            keywords = {'return', 'if', 'while', 'for', 'switch', 'case', 'else', 'do'} # Require a postfix callable expression at this position.
            if identifier and identifier.group(1) not in keywords: # Include a constructor or function name before its call.
                start = primary_expression_start(code, start) # Preserve the complete call as the length receiver.
            elif prefix[-1] in ')]': # A postfix call can follow an indexed or parenthesized receiver.
                start = primary_expression_start(code, start) # Include that earlier postfix expression as well.
    else: # A simple receiver ends with an identifier.
        identifier = re.search(r'\b[A-Za-z_]\w*$', code[:end]) # Select the identifier ending at this member boundary.
        if not identifier: # Require an array-valued receiver for the length query.
            raise ShaderError('unsupported array length receiver') # Explain the receiver boundary validation error.
        start = identifier.start() # Keep the original source offset of this receiver token.
    prefix = code[:start].rstrip() # Check whether this token belongs to a longer member chain.
    if prefix.endswith('.'): # Named storage-block fields are part of their block receiver.
        start = primary_expression_start(code, len(prefix) - 1) # Include that preceding block or nested member expression.
    return start # Return the first character of the complete postfix receiver.


def statement_end(source, position): # Locate the complete statement including nested control flow.
    """Locate a structured statement's end for lexical for-initializer scope."""
    while position < len(source) and source[position].isspace(): # Continue while the parser still has matching input.
        position += 1 # Compute position for the following operation.

    def close(opening, left, right): # Find the matching end of this bracketed expression.
        depth = 0 # Compute depth for the following operation.
        for i in range(opening, len(source)): # Process each i in this sequence.
            if source[i] == left: # Select this branch when its condition holds.
                depth += 1 # Compute depth for the following operation.
            elif source[i] == right: # Try this alternative condition after earlier branches fail.
                depth -= 1 # Supply depth for the continued expression.
                if not depth: # Select this branch when its condition holds.
                    return i + 1 # Return this computed value to the caller.
        raise ShaderError('unterminated shader statement') # Report the shader declaration validation error.

    if position < len(source) and source[position] == '{': # Select this branch when its condition holds.
        return close(position, '{', '}') # Return this computed value to the caller.
    control = re.match(r'(for|while|if|switch)\s*\(', source[position:]) # Compute control for the following operation.
    if control: # Select this branch when its condition holds.
        opening = position + control.end() - 1 # Compute opening for the following operation.
        end = statement_end(source, close(opening, '(', ')')) # Compute end for the following operation.
        tail = source[end:] # Compute tail for the following operation.
        alternate = re.match(r'\s*else\b', tail) if control.group(1) == 'if' else None # Compute alternate for the following operation.
        return statement_end(source, end + alternate.end()) if alternate else end # Return this computed value to the caller.
    if re.match(r'do\b', source[position:]): # Select this branch when its condition holds.
        end = statement_end(source, position + 2) # Compute end for the following operation.
        tail = re.match(r'\s*while\s*\(', source[end:]) # Compute tail for the following operation.
        if not tail: # Select this branch when its condition holds.
            raise ShaderError('malformed do/while statement') # Report the shader declaration validation error.
        end = close(end + tail.end() - 1, '(', ')') # Compute end for the following operation.
        while end < len(source) and source[end].isspace(): # Continue while the parser still has matching input.
            end += 1 # Compute end for the following operation.
        return end + 1 # Return this computed value to the caller.
    parentheses = brackets = 0 # Compute parentheses for the following operation.
    for i in range(position, len(source)): # Process each i in this sequence.
        token = source[i] # Compute token for the following operation.
        if token == '(': # Select this branch when its condition holds.
            parentheses += 1 # Compute parentheses for the following operation.
        elif token == ')': # Try this alternative condition after earlier branches fail.
            parentheses -= 1 # Supply parentheses for the continued expression.
        elif token == '[': # Try this alternative condition after earlier branches fail.
            brackets += 1 # Compute brackets for the following operation.
        elif token == ']': # Try this alternative condition after earlier branches fail.
            brackets -= 1 # Supply brackets for the continued expression.
        elif token == ';' and not parentheses and not brackets: # Try this alternative condition after earlier branches fail.
            return i + 1 # Return this computed value to the caller.
    raise ShaderError('unterminated shader statement') # Report the shader declaration validation error.


def scoped_integer_constants(source, definitions, position=None): # Resolve dimension aliases within their lexical statement scope.
    """Resolve integer declarations in their lexical scope.

    A top-level query returns constants for interface metadata. A position query
    uses the enclosing scopes and respects local variable shadows.
    """
    scopes = [dict(definitions)] # Compute scopes for the following operation.
    declarations = r'\b(const\s+)?(?:uint|int)\s+(\w+)\s*(?:=\s*([^;{}]+))?;' # Compute declarations for the following operation.
    tokens = re.compile(r'\bfor\s*\(|[{}]|' + declarations) # Compute tokens for the following operation.
    loops = [] # Compute loops for the following operation.
    for token in tokens.finditer(source if position is None else source[:position]): # Process each token in this sequence.
        while loops and loops[-1] <= token.start(): # Continue while the parser still has matching input.
            loops.pop() # Apply pop to these arguments.
            scopes.pop() # Apply pop to these arguments.
        if re.match(r'for\b', token.group()): # Select this branch when its condition holds.
            loops.append(statement_end(source, token.start())) # Apply append to these arguments.
            scopes.append({}) # Apply append to these arguments.
        elif token.group() == '{': # Try this alternative condition after earlier branches fail.
            scopes.append({}) # Apply append to these arguments.
        elif token.group() == '}': # Try this alternative condition after earlier branches fail.
            if len(scopes) > 1: # Select this branch when its condition holds.
                scopes.pop() # Apply pop to these arguments.
        elif position is not None or len(scopes) == 1: # Try this alternative condition after earlier branches fail.
            qualifier, name, expression = token.groups() # Apply groups to these arguments.
            scopes[-1][name] = expression.strip() if qualifier and expression else None # Apply strip to these arguments.
    boundary = len(source) if position is None else position # Compute boundary for the following operation.
    while loops and loops[-1] <= boundary: # Continue while the parser still has matching input.
        loops.pop() # Apply pop to these arguments.
        scopes.pop() # Apply pop to these arguments.
    result = {} # Compute result for the following operation.
    for scope in scopes: # Process each scope in this sequence.
        result.update(scope) # Apply update to these arguments.
    return result # Return this computed value to the caller.


def separate_array_declarators(source, types): # Give each array declarator its own C++ wrapper declaration.
    """Give comma-separated private arrays their own C++ wrapper type declarations."""
    code = mask_noncode(source) # Compute code for the following operation.
    edits = [] # Compute edits for the following operation.
    pattern = re.compile(r'\b(' + types + r')\s+(\w+)\s*(?=[\[=,;])') # Compute pattern for the following operation.
    for match in pattern.finditer(code): # Process each match in this sequence.
        begin = match.start(2) # Compute begin for the following operation.
        parentheses = brackets = 0 # Compute parentheses for the following operation.
        cuts = [begin] # Compute cuts for the following operation.
        end = None # Compute end for the following operation.
        for position in range(begin, len(code)): # Process each position in this sequence.
            token = code[position] # Compute token for the following operation.
            if token == '{' or token == '}': # Select this branch when its condition holds.
                break # Finish this scan at the current boundary.
            if token == '(': # Select this branch when its condition holds.
                parentheses += 1 # Compute parentheses for the following operation.
            elif token == ')': # Try this alternative condition after earlier branches fail.
                if not parentheses: # Select this branch when its condition holds.
                    break # Finish this scan at the current boundary.
                parentheses -= 1 # Supply parentheses for the continued expression.
            elif token == '[': # Try this alternative condition after earlier branches fail.
                brackets += 1 # Compute brackets for the following operation.
            elif token == ']': # Try this alternative condition after earlier branches fail.
                brackets -= 1 # Supply brackets for the continued expression.
            elif not parentheses and not brackets: # Try this alternative condition after earlier branches fail.
                if token == ',': # Select this branch when its condition holds.
                    cuts.append(position + 1) # Apply append to these arguments.
                elif token == ';': # Try this alternative condition after earlier branches fail.
                    end = position # Compute end for the following operation.
                    break # Finish this scan at the current boundary.
        if end is None or len(cuts) < 2: # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        stops = [p - 1 for p in cuts[1:]] + [end] # Compute stops for the following operation.
        declarators = [source[a:b].strip() for a, b in zip(cuts, stops)] # Compute declarators for the following operation.
        if not any(re.match(r'\w+\s*\[', d) for d in declarators): # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        start = match.start() # Compute start for the following operation.
        qualifier = re.search(r'\bconst\s*$', code[:start]) # Compute qualifier for the following operation.
        prefix = '' # Compute prefix for the following operation.
        if qualifier: # Select this branch when its condition holds.
            start = qualifier.start() # Compute start for the following operation.
            prefix = 'const ' # Compute prefix for the following operation.
        edits.append( # Apply append to these arguments.
            ( # Finish this expression or collection.
                start, # Supply start for the continued expression.
                end + 1, # Supply end for the continued expression.
                ' '.join(prefix + match.group(1) + ' ' + d + ';' for d in declarators), # Apply join to these arguments.
            ) # Finish this expression or collection.
        ) # Finish this expression or collection.
    for start, end, replacement in reversed(edits): # Process each start, end, replacement in this sequence.
        source = source[:start] + replacement + source[end:] # Compute source for the following operation.
    return source # Return this computed value to the caller.


def portable_gpu_declarations(source): # Preserve desktop initialization semantics under GLES declaration rules.
    """Keep declaration values while making desktop initialization valid in GLES."""
    code = mask_noncode(source) # Compute code for the following operation.
    types = '|'.join(sorted(UNIFORMS, key=len, reverse=True)) # Compute types for the following operation.
    function = r'(?P<function>\b(?:void|' + types + r')\s+\w+\s*\([^;{}]*\)\s*\{)' # Compute function for the following operation.
    declaration = ( # Compute declaration for the following operation.
        r'(?P<declaration>\b(?P<qualifiers>(?:(?:const|highp|mediump|lowp)\s+)*)(?P<type>'
        + types # Supply types for the continued expression.
        + r')\s+(?=[A-Za-z_]\w*\s*(?:\[|=|,|;)))'
    ) # Finish this expression or collection.
    events = re.compile( # Compute events for the following operation.
        function + r'|(?P<loop>\bfor\s*\()|(?P<brace>[{}])|' + declaration # Supply function for the continued expression.
    ) # Finish this expression or collection.
    definitions = { # Compute definitions for the following operation.
        m.group(1): m.group(2).strip() # Apply group to these arguments.
        for m in re.finditer(r'(?m)^[ \t]*#\s*define\s+(\w+)\s+([^\n]+)', code) # Process each m in this sequence.
    } # Finish this expression or collection.
    scopes = [dict(definitions, gl_WorkGroupSize=True)] # Compute scopes for the following operation.
    loops, edits = [], [] # Supply loops for the continued expression.
    # Only these GLSL builtins produce constant expressions from constant operands.
    pure_calls = UNIFORMS | { # Compute pure calls for the following operation.
        'radians',
        'degrees',
        'sin',
        'cos',
        'tan',
        'asin',
        'acos',
        'atan',
        'sinh',
        'cosh',
        'tanh',
        'asinh',
        'acosh',
        'atanh',
        'pow',
        'exp',
        'log',
        'exp2',
        'log2',
        'sqrt',
        'inversesqrt',
        'abs',
        'sign',
        'floor',
        'trunc',
        'round',
        'roundEven',
        'ceil',
        'fract',
        'mod',
        'min',
        'max',
        'clamp',
        'mix',
        'step',
        'smoothstep',
        'fma',
        'ldexp',
        'length',
        'distance',
        'dot',
        'cross',
        'normalize',
        'faceforward',
        'reflect',
        'refract',
        'matrixCompMult',
        'outerProduct',
        'transpose',
        'determinant',
        'inverse',
        'lessThan',
        'lessThanEqual',
        'greaterThan',
        'greaterThanEqual',
        'equal',
        'notEqual',
        'any',
        'all',
        'not',
    } # Finish this expression or collection.

    def constant(expression, environment, seen=()): # Distinguish constant expressions from runtime local values.
        for token in re.finditer(r'\b[A-Za-z_]\w*\b', expression): # Process each token in this sequence.
            name = token.group() # Compute name for the following operation.
            if expression[: token.start()].rstrip().endswith('.'): # Select this branch when its condition holds.
                continue # Skip the remaining work for this iteration.
            if name in pure_calls or name in {'true', 'false'}: # Select this branch when its condition holds.
                continue # Skip the remaining work for this iteration.
            value = environment.get(name) # Compute value for the following operation.
            if value is True: # Select this branch when its condition holds.
                continue # Skip the remaining work for this iteration.
            if ( # Select this branch when its condition holds.
                not value # Require a resolved constant value for this expression.
                or name in seen # Supply or for the continued expression.
                or not constant(value, environment, seen + (name,)) # Apply constant to these arguments.
            ): # Finish this expression or collection.
                return False # Return this computed value to the caller.
        return True # Return this computed value to the caller.

    position = 0 # Compute position for the following operation.
    while True: # Continue while the parser still has matching input.
        event = events.search(code, position) # Compute event for the following operation.
        if not event: # Select this branch when its condition holds.
            break # Finish this scan at the current boundary.
        while loops and loops[-1] <= event.start(): # Continue while the parser still has matching input.
            loops.pop() # Apply pop to these arguments.
            scopes.pop() # Apply pop to these arguments.
        position = event.end() # Compute position for the following operation.
        line_start = code.rfind('\n', 0, event.start()) + 1 # Compute line start for the following operation.
        if code[line_start : event.start()].lstrip().startswith('#'): # Select this branch when its condition holds.
            position = code.find('\n', event.start()) # Compute position for the following operation.
            if position < 0: # Select this branch when its condition holds.
                break # Finish this scan at the current boundary.
            continue # Skip the remaining work for this iteration.
        if event.group('function'): # Select this branch when its condition holds.
            signature = event.group() # Compute signature for the following operation.
            arguments = signature[signature.index('(') + 1 : signature.rfind(')')] # Compute arguments for the following operation.
            parameters = re.findall(r'\b(?:' + types + r')\s+(\w+)', arguments) # Compute parameters for the following operation.
            scopes.append({name: None for name in parameters}) # Apply append to these arguments.
            continue # Skip the remaining work for this iteration.
        if event.group('loop'): # Select this branch when its condition holds.
            loops.append(statement_end(code, event.start())) # Apply append to these arguments.
            scopes.append({}) # Apply append to these arguments.
            continue # Skip the remaining work for this iteration.
        if event.group('brace'): # Select this branch when its condition holds.
            if event.group() == '{': # Select this branch when its condition holds.
                scopes.append({}) # Apply append to these arguments.
            elif len(scopes) > 1: # Try this alternative condition after earlier branches fail.
                scopes.pop() # Apply pop to these arguments.
            continue # Skip the remaining work for this iteration.
        end = statement_end(code, event.start()) - 1 # Compute end for the following operation.
        # Split commas only outside constructors and array dimensions.
        boundaries, depth, start = [], 0, event.end() # Apply end to these arguments.
        for i in range(start, end): # Process each i in this sequence.
            if code[i] in '([': # Select this branch when its condition holds.
                depth += 1 # Compute depth for the following operation.
            elif code[i] in ')]': # Try this alternative condition after earlier branches fail.
                depth -= 1 # Supply depth for the continued expression.
            elif code[i] == ',' and depth == 0: # Try this alternative condition after earlier branches fail.
                boundaries.append((start, i)) # Apply append to these arguments.
                start = i + 1 # Compute start for the following operation.
        boundaries.append((start, end)) # Apply append to these arguments.
        environment = {} # Compute environment for the following operation.
        for scope in scopes: # Process each scope in this sequence.
            environment.update(scope) # Apply update to these arguments.
        parts = [] # Compute parts for the following operation.
        is_const = 'const' in event.group('qualifiers').split() # Compute is const for the following operation.
        local = len(scopes) > 1 # Compute local for the following operation.
        for start, stop in boundaries: # Process each start, stop in this sequence.
            item = re.match( # Compute item for the following operation.
                r'\s*(\w+)\s*((?:\[[^]]*\]\s*)*)(?:=\s*([\s\S]*))?$', code[start:stop] # Supply r for the continued expression.
            ) # Finish this expression or collection.
            if not item: # Select this branch when its condition holds.
                break # Finish this scan at the current boundary.
            name, array, expression = item.groups() # Apply groups to these arguments.
            known = bool( # Compute known for the following operation.
                is_const # Supply is const for the continued expression.
                and expression is not None # Supply and for the continued expression.
                and constant(expression, environment) # Apply constant to these arguments.
            ) # Finish this expression or collection.
            environment[name] = expression if known else None # Supply environment for the continued expression.
            scopes[-1][name] = environment[name] # Supply scopes for the continued expression.
            parts.append((start, stop, item, array, expression, known)) # Apply append to these arguments.
        else: # Handle the remaining case.
            remove_const = is_const and local and any(not part[-1] for part in parts) # Compute remove const for the following operation.
            # One const qualifier applies to the complete for-initializer declaration.
            # Preserve constant dimension aliases when other declarators are runtime values.
            for_initializer = bool( # Compute for initializer for the following operation.
                loops # Supply loops for the continued expression.
                and code[code.rfind('for', 0, event.start()) : event.start()].find(';') # Apply rfind to these arguments.
                < 0 # Supply this value for the continued expression.
            ) # Finish this expression or collection.
            mixed = remove_const and any(part[-1] for part in parts) # Compute mixed for the following operation.
            if mixed and for_initializer: # Select this branch when its condition holds.
                raise ShaderError( # Report the shader declaration validation error.
                    'mixed constant/runtime for initializers require separate declarations'
                ) # Finish this expression or collection.
            if mixed: # Select this branch when its condition holds.
                declarations = [] # Compute declarations for the following operation.
                for start, stop, item, array, expression, known in parts: # Process each start, stop, item, array, expression, known in this sequence.
                    prefix = event.group('qualifiers') # Compute prefix for the following operation.
                    if not known: # Select this branch when its condition holds.
                        prefix = re.sub(r'\bconst\s*', '', prefix) # Compute prefix for the following operation.
                    body = source[start:stop].strip() # Compute body for the following operation.
                    if ( # Select this branch when its condition holds.
                        expression is not None # Supply expression for the continued expression.
                        and not array # Supply and for the continued expression.
                        and event.group('type') in UNIFORMS # Apply group to these arguments.
                    ): # Finish this expression or collection.
                        offset = item.start(3) # Compute offset for the following operation.
                        body = ( # Compute body for the following operation.
                            source[start : start + offset].strip() # Apply strip to these arguments.
                            + event.group('type') # Apply group to these arguments.
                            + '('
                            + source[start + offset : stop].strip() # Apply strip to these arguments.
                            + ')'
                        ) # Finish this expression or collection.
                    declarations.append(prefix + event.group('type') + ' ' + body + ';') # Apply append to these arguments.
                edits.append((event.start(), end + 1, '\n'.join(declarations))) # Apply append to these arguments.
            else: # Handle the remaining case.
                if remove_const: # Select this branch when its condition holds.
                    qualifier = re.search(r'\bconst\b', event.group('qualifiers')) # Compute qualifier for the following operation.
                    edits.append( # Apply append to these arguments.
                        ( # Finish this expression or collection.
                            event.start() + qualifier.start(), # Apply start to these arguments.
                            event.start() + qualifier.end(), # Apply start to these arguments.
                            '',
                        ) # Finish this expression or collection.
                    ) # Finish this expression or collection.
                for start, stop, item, array, expression, known in parts: # Process each start, stop, item, array, expression, known in this sequence.
                    if ( # Select this branch when its condition holds.
                        expression is not None # Supply expression for the continued expression.
                        and not array # Supply and for the continued expression.
                        and event.group('type') in UNIFORMS # Apply group to these arguments.
                    ): # Finish this expression or collection.
                        first = start + item.start(3) # Compute first for the following operation.
                        edits.append( # Apply append to these arguments.
                            ( # Finish this expression or collection.
                                first, # Supply first for the continued expression.
                                stop, # Supply stop for the continued expression.
                                event.group('type') + '(' + source[first:stop] + ')', # Apply group to these arguments.
                            ) # Finish this expression or collection.
                        ) # Finish this expression or collection.
            position = end + 1 # Compute position for the following operation.
            continue # Skip the remaining work for this iteration.
        position = end + 1 # Compute position for the following operation.
    for start, end, replacement in sorted(edits, reverse=True): # Process each start, end, replacement in this sequence.
        source = source[:start] + replacement + source[end:] # Compute source for the following operation.
    return source # Return this computed value to the caller.


def portable_gpu_promotions(source): # Make the desktop numeric promotions explicit for GLES.
    """Spell out desktop numeric promotions for the stricter GLES type rules."""
    code = mask_noncode(source) # Compute code for the following operation.
    code = re.sub(r'(?m)^[ \t]*#[^\n]*', lambda m: ' ' * len(m.group()), code) # Compute code for the following operation.
    kinds = '|'.join(sorted(UNIFORMS, key=len, reverse=True)) # Compute kinds for the following operation.
    numeric = re.compile(r'^(float|int|uint|bool|[iub]?vec[234]|mat[234](?:x[234])?)$') # Compute numeric for the following operation.
    symbols = {} # Compute symbols for the following operation.
    fields = {} # Compute fields for the following operation.
    functions = {} # Compute functions for the following operation.

    def remember(name, kind): # Record every declared type for this identifier.
        symbols.setdefault(name, set()).add(kind) # Apply setdefault to these arguments.

    for declaration in re.finditer( # Process each declaration in this sequence.
        r'\b(' + kinds + r')\s+([A-Za-z_]\w*)\s*(\[)?', code # Apply b to these arguments.
    ): # Finish this expression or collection.
        remember( # Apply remember to these arguments.
            declaration.group(2), # Apply group to these arguments.
            declaration.group(1) + ('[]' if declaration.group(3) else ''), # Apply group to these arguments.
        ) # Finish this expression or collection.
    for structure in re.finditer(r'\bstruct\s+(\w+)\s*\{([^{}]*)\}', code): # Process each structure in this sequence.
        fields[structure.group(1)] = { # Apply group to these arguments.
            name: kind # Supply name for the continued expression.
            for kind, name in re.findall( # Process each kind, name in this sequence.
                r'\b(' + kinds + r')\s+(\w+)\s*;', structure.group(2) # Apply b to these arguments.
            ) # Finish this expression or collection.
        } # Finish this expression or collection.
        for instance in re.finditer( # Process each instance in this sequence.
            r'\b' + structure.group(1) + r'\s+(\w+)\s*[;=]', code # Apply group to these arguments.
        ): # Finish this expression or collection.
            remember(instance.group(1), structure.group(1)) # Apply remember to these arguments.
    signature = re.compile( # Compute signature for the following operation.
        r'\b(void|' + kinds + r')\s+(\w+)\s*\(([^;{}]*)\)\s*(?=[{;])' # Apply b to these arguments.
    ) # Finish this expression or collection.
    for match in signature.finditer(code): # Process each match in this sequence.
        parameters = [] # Compute parameters for the following operation.
        for parameter in match.group(3).split(','): # Process each parameter in this sequence.
            value = re.search(r'\b(' + kinds + r')\s+(\w+)\s*(\[)?', parameter) # Compute value for the following operation.
            if value: # Select this branch when its condition holds.
                parameters.append( # Apply append to these arguments.
                    (value.group(1), bool(re.search(r'\b(?:out|inout)\b', parameter))) # Apply group to these arguments.
                ) # Finish this expression or collection.
        functions.setdefault(match.group(2), set()).add( # Apply setdefault to these arguments.
            (match.group(1), tuple(parameters)) # Apply group to these arguments.
        ) # Finish this expression or collection.
    symbols.update( # Apply update to these arguments.
        { # Finish this expression or collection.
            name: {kind} # Supply name for the continued expression.
            for name, kind in { # Process each name, kind in this sequence.
                'gl_GlobalInvocationID': 'uvec3', # Supply gl GlobalInvocationID for the continued expression.
                'gl_LocalInvocationID': 'uvec3', # Supply gl LocalInvocationID for the continued expression.
                'gl_WorkGroupID': 'uvec3', # Supply gl WorkGroupID for the continued expression.
                'gl_NumWorkGroups': 'uvec3', # Supply gl NumWorkGroups for the continued expression.
                'gl_WorkGroupSize': 'uvec3', # Supply gl WorkGroupSize for the continued expression.
                'gl_LocalInvocationIndex': 'uint', # Supply gl LocalInvocationIndex for the continued expression.
            }.items() # Apply items to these arguments.
        } # Finish this expression or collection.
    ) # Finish this expression or collection.
    token_pattern = r'(?:0[xX][\da-fA-F]+[uU]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuU]?)|[A-Za-z_]\w*|(?:<<=|>>=|\+\+|--|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=)|[^\s]' # Compute token pattern for the following operation.
    tokens = [(m.group(), m.start(), m.end()) for m in re.finditer(token_pattern, code)] # Compute tokens for the following operation.
    priorities = { # Compute priorities for the following operation.
        '=': 1, # Supply this value for the continued expression.
        '+=': 1, # Supply this value for the continued expression.
        '-=': 1, # Supply this value for the continued expression.
        '*=': 1, # Supply this value for the continued expression.
        '/=': 1, # Supply this value for the continued expression.
        '%=': 1, # Supply this value for the continued expression.
        '&=': 1, # Supply this value for the continued expression.
        '|=': 1, # Supply this value for the continued expression.
        '^=': 1, # Supply this value for the continued expression.
        '<<=': 1, # Supply this value for the continued expression.
        '>>=': 1, # Supply this value for the continued expression.
        '?': 2, # Supply this value for the continued expression.
        '||': 3, # Supply this value for the continued expression.
        '&&': 4, # Supply this value for the continued expression.
        '|': 5, # Supply this value for the continued expression.
        '^': 6, # Supply this value for the continued expression.
        '&': 7, # Supply this value for the continued expression.
        '==': 8, # Supply this value for the continued expression.
        '!=': 8, # Supply this value for the continued expression.
        '<': 9, # Supply this value for the continued expression.
        '>': 9, # Supply this value for the continued expression.
        '<=': 9, # Supply this value for the continued expression.
        '>=': 9, # Supply this value for the continued expression.
        '<<': 10, # Supply this value for the continued expression.
        '>>': 10, # Supply this value for the continued expression.
        '+': 11, # Supply this value for the continued expression.
        '-': 11, # Supply this value for the continued expression.
        '*': 12, # Supply this value for the continued expression.
        '/': 12, # Supply this value for the continued expression.
        '%': 12, # Supply this value for the continued expression.
    } # Finish this expression or collection.
    conversions = set() # Compute conversions for the following operation.

    def scalar(kind): # Find the component type of a numeric scalar or vector.
        if not kind or not numeric.fullmatch(kind): # Select this branch when its condition holds.
            return None # Return this computed value to the caller.
        if kind.startswith('mat') or kind.startswith('vec'): # Select this branch when its condition holds.
            return 'float' # Return this computed value to the caller.
        if kind.startswith('ivec'): # Select this branch when its condition holds.
            return 'int' # Return this computed value to the caller.
        if kind.startswith('uvec'): # Select this branch when its condition holds.
            return 'uint' # Return this computed value to the caller.
        if kind.startswith('bvec'): # Select this branch when its condition holds.
            return 'bool' # Return this computed value to the caller.
        return kind # Return this computed value to the caller.

    def shape(kind): # Keep the numeric dimensions separate from the component type.
        if not kind: # Select this branch when its condition holds.
            return None # Return this computed value to the caller.
        return ( # Return this computed value to the caller.
            kind[kind.index('vec') :] # Apply index to these arguments.
            if 'vec' in kind # Select this branch when its condition holds.
            else kind if kind.startswith('mat') else '' # Apply startswith to these arguments.
        ) # Finish this expression or collection.

    def replace_scalar(kind, base): # Retain the vector shape while changing its component type.
        dimensions = shape(kind) # Compute dimensions for the following operation.
        if dimensions and dimensions.startswith('vec'): # Select this branch when its condition holds.
            return ( # Return this computed value to the caller.
                '' if base == 'float' else 'i' if base == 'int' else 'u' # Supply if for the continued expression.
            ) + dimensions # Supply dimensions for the continued expression.
        return base if dimensions == '' else kind # Return this computed value to the caller.

    def promote(left, right): # Find the common desktop type for these two numeric operands.
        a, b = scalar(left), scalar(right) # Apply scalar to these arguments.
        if a not in {'float', 'int', 'uint'} or b not in {'float', 'int', 'uint'}: # Select this branch when its condition holds.
            return None, None, None # Return this computed value to the caller.
        if shape(left) and shape(right) and shape(left) != shape(right): # Select this branch when its condition holds.
            return None, None, None # Return this computed value to the caller.
        base = 'float' if 'float' in {a, b} else 'uint' if 'uint' in {a, b} else 'int' # Compute base for the following operation.
        ltype, rtype = replace_scalar(left, base), replace_scalar(right, base) # Apply replace scalar to these arguments.
        return ltype, rtype, ltype if shape(left) else rtype # Return this computed value to the caller.

    def cast(node, kind): # Request a conversion only when the known component types differ.
        if ( # Select this branch when its condition holds.
            node # Supply node for the continued expression.
            and kind # Supply and for the continued expression.
            and node[2] # Supply and for the continued expression.
            and node[2] != kind # Supply and for the continued expression.
            and scalar(node[2]) in {'float', 'int', 'uint'} # Apply scalar to these arguments.
            and scalar(kind) in {'float', 'int', 'uint'} # Apply scalar to these arguments.
        ): # Finish this expression or collection.
            if shape(node[2]) == shape(kind): # Select this branch when its condition holds.
                conversions.add((node[0], node[1], kind)) # Apply add to these arguments.

    def symbol(name): # Use an identifier type only when its declarations agree.
        candidates = symbols.get(name, set()) # Compute candidates for the following operation.
        return next(iter(candidates)) if len(candidates) == 1 else None # Return this computed value to the caller.

    class Expression: # Group these parser operations or diagnostic values.
        def __init__(self, position): # Start parsing at this token with a bounded recursion depth.
            self.position = position # Compute self.position for the following operation.
            self.depth = 0 # Compute self.depth for the following operation.

        def value(self): # Inspect the next token at the current parser position.
            return tokens[self.position][0] if self.position < len(tokens) else '' # Return this computed value to the caller.

        def parse(self, minimum=1): # Apply GLSL operator precedence while recording explicit promotions.
            self.depth += 1 # Compute self.depth for the following operation.
            if self.depth > 128: # Select this branch when its condition holds.
                self.depth -= 1 # Supply self for the continued expression.
                return None # Return this computed value to the caller.
            node = self.atom() # Compute node for the following operation.
            while node: # Continue while the parser still has matching input.
                operation = self.value() # Compute operation for the following operation.
                priority = priorities.get(operation, 0) # Compute priority for the following operation.
                if priority < minimum: # Select this branch when its condition holds.
                    break # Finish this scan at the current boundary.
                self.position += 1 # Compute self.position for the following operation.
                if operation == '?': # Select this branch when its condition holds.
                    yes = self.parse() # Compute yes for the following operation.
                    if self.value() != ':': # Select this branch when its condition holds.
                        break # Finish this scan at the current boundary.
                    self.position += 1 # Compute self.position for the following operation.
                    no = self.parse(priority) # Parse the conditional expression's false branch.
                    if not yes or not no: # Select this branch when its condition holds.
                        break # Finish this scan at the current boundary.
                    a, b, result = promote(yes[2], no[2]) # Apply promote to these arguments.
                    cast(yes, a) # Apply cast to these arguments.
                    cast(no, b) # Apply cast to these arguments.
                    node = ( # Compute node for the following operation.
                        node[0], # Supply node for the continued expression.
                        no[1], # Use the false branch's ending source position.
                        result or (yes[2] if yes[2] == no[2] else None), # Apply or to these arguments.
                    ) # Finish this expression or collection.
                    continue # Skip the remaining work for this iteration.
                right = self.parse(priority if priority == 1 else priority + 1) # Compute right for the following operation.
                if not right: # Select this branch when its condition holds.
                    break # Finish this scan at the current boundary.
                if operation == '=': # Select this branch when its condition holds.
                    cast(right, node[2]) # Apply cast to these arguments.
                    result = node[2] # Compute result for the following operation.
                elif operation.endswith('=') and priority == 1: # Try this alternative condition after earlier branches fail.
                    # Compound assignment already converts its result to the left-hand type.
                    _, b, _ = promote(node[2], right[2]) # Apply promote to these arguments.
                    cast(right, b) # Apply cast to these arguments.
                    result = node[2] # Compute result for the following operation.
                elif operation in {'<<', '>>'}: # Try this alternative condition after earlier branches fail.
                    result = node[2] # Compute result for the following operation.
                elif operation in {'&&', '||'}: # Try this alternative condition after earlier branches fail.
                    result = 'bool' # Compute result for the following operation.
                else: # Handle the remaining case.
                    a, b, result = promote(node[2], right[2]) # Apply promote to these arguments.
                    cast(node, a) # Apply cast to these arguments.
                    cast(right, b) # Apply cast to these arguments.
                    if priority in {8, 9}: # Select this branch when its condition holds.
                        result = 'bool' # Compute result for the following operation.
                node = (node[0], right[1], result) # Compute node for the following operation.
            self.depth -= 1 # Supply self for the continued expression.
            return node # Return this computed value to the caller.

        def atom(self): # Read one literal, identifier, call or indexed member expression.
            if self.position >= len(tokens): # Select this branch when its condition holds.
                return None # Return this computed value to the caller.
            value, start, end = tokens[self.position] # Supply value for the continued expression.
            if value in {'+', '-', '!', '~', '++', '--'}: # Select this branch when its condition holds.
                self.position += 1 # Compute self.position for the following operation.
                child = self.atom() # Compute child for the following operation.
                return ( # Return this computed value to the caller.
                    (start, child[1], 'bool' if value == '!' else child[2]) # Supply start for the continued expression.
                    if child # Select this branch when its condition holds.
                    else None # Supply else for the continued expression.
                ) # Finish this expression or collection.
            if value == '(': # Select this branch when its condition holds.
                self.position += 1 # Compute self.position for the following operation.
                child = self.parse() # Compute child for the following operation.
                if not child or self.value() != ')': # Select this branch when its condition holds.
                    return None # Return this computed value to the caller.
                end = tokens[self.position][2] # Compute end for the following operation.
                self.position += 1 # Compute self.position for the following operation.
                node = (start, end, child[2]) # Compute node for the following operation.
            elif re.fullmatch( # Try this alternative condition after earlier branches fail.
                r'(?:0[xX][\da-fA-F]+[uU]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuU]?)',
                value, # Supply value for the continued expression.
            ): # Finish this expression or collection.
                kind = ( # Compute kind for the following operation.
                    'uint'
                    if value[-1:] in {'u', 'U'} # Select this branch when its condition holds.
                    else ( # Apply else to these arguments.
                        'float'
                        if not value.lower().startswith('0x') # Select this branch when its condition holds.
                        and re.search(r'[.eEfF]', value) # Apply search to these arguments.
                        else 'int' # Supply else for the continued expression.
                    ) # Finish this expression or collection.
                ) # Finish this expression or collection.
                self.position += 1 # Compute self.position for the following operation.
                node = (start, end, kind) # Compute node for the following operation.
            elif re.fullmatch(r'[A-Za-z_]\w*', value): # Try this alternative condition after earlier branches fail.
                self.position += 1 # Compute self.position for the following operation.
                kind = 'bool' if value in {'true', 'false'} else symbol(value) # Compute kind for the following operation.
                node = (start, end, kind) # Compute node for the following operation.
                if self.value() == '(': # Select this branch when its condition holds.
                    self.position += 1 # Compute self.position for the following operation.
                    arguments = [] # Compute arguments for the following operation.
                    while self.position < len(tokens) and self.value() != ')': # Continue while the parser still has matching input.
                        argument = self.parse() # Compute argument for the following operation.
                        if not argument: # Select this branch when its condition holds.
                            return None # Return this computed value to the caller.
                        arguments.append(argument) # Apply append to these arguments.
                        if self.value() != ',': # Select this branch when its condition holds.
                            break # Finish this scan at the current boundary.
                        self.position += 1 # Compute self.position for the following operation.
                    if self.value() != ')': # Select this branch when its condition holds.
                        return None # Return this computed value to the caller.
                    end = tokens[self.position][2] # Compute end for the following operation.
                    self.position += 1 # Compute self.position for the following operation.
                    if value in UNIFORMS: # Select this branch when its condition holds.
                        kind = value # Compute kind for the following operation.
                    elif value in functions and len(functions[value]) == 1: # Try this alternative condition after earlier branches fail.
                        kind, parameters = next(iter(functions[value])) # Apply next to these arguments.
                        if len(parameters) == len(arguments): # Select this branch when its condition holds.
                            for argument, (expected, output) in zip( # Process each argument, (expected, output) in this sequence.
                                arguments, parameters # Supply arguments for the continued expression.
                            ): # Finish this expression or collection.
                                if not output: # Select this branch when its condition holds.
                                    cast(argument, expected) # Apply cast to these arguments.
                    elif value in {'dot', 'length', 'distance', 'determinant'}: # Try this alternative condition after earlier branches fail.
                        kind = 'float' # Compute kind for the following operation.
                    elif ( # Try this alternative condition after earlier branches fail.
                        value # Supply value for the continued expression.
                        in { # Supply in for the continued expression.
                            'lessThan',
                            'lessThanEqual',
                            'greaterThan',
                            'greaterThanEqual',
                            'equal',
                            'notEqual',
                        } # Finish this expression or collection.
                        and arguments # Supply and for the continued expression.
                    ): # Finish this expression or collection.
                        kind = ( # Compute kind for the following operation.
                            'b' + shape(arguments[0][2]) # Apply shape to these arguments.
                            if shape(arguments[0][2]) # Select this branch when its condition holds.
                            else None # Supply else for the continued expression.
                        ) # Finish this expression or collection.
                    elif value in {'any', 'all'}: # Try this alternative condition after earlier branches fail.
                        kind = 'bool' # Compute kind for the following operation.
                    elif value in {'imageLoad'}: # Try this alternative condition after earlier branches fail.
                        kind = None # Compute kind for the following operation.
                    elif arguments and value in { # Try this alternative condition after earlier branches fail.
                        'radians',
                        'degrees',
                        'sin',
                        'cos',
                        'tan',
                        'asin',
                        'acos',
                        'atan',
                        'sinh',
                        'cosh',
                        'tanh',
                        'asinh',
                        'acosh',
                        'atanh',
                        'pow',
                        'exp',
                        'log',
                        'exp2',
                        'log2',
                        'sqrt',
                        'inversesqrt',
                        'abs',
                        'sign',
                        'floor',
                        'trunc',
                        'round',
                        'roundEven',
                        'ceil',
                        'fract',
                        'mod',
                        'min',
                        'max',
                        'clamp',
                        'mix',
                        'step',
                        'smoothstep',
                        'fma',
                        'ldexp',
                        'normalize',
                        'faceforward',
                        'reflect',
                        'refract',
                        'matrixCompMult',
                        'transpose',
                        'inverse',
                    }: # Finish this expression or collection.
                        kind = arguments[0][2] # Compute kind for the following operation.
                    else: # Handle the remaining case.
                        kind = None # Compute kind for the following operation.
                    node = (start, end, kind) # Compute node for the following operation.
            else: # Handle the remaining case.
                return None # Return this computed value to the caller.
            while self.position < len(tokens): # Continue while the parser still has matching input.
                value = self.value() # Compute value for the following operation.
                if value == '[': # Select this branch when its condition holds.
                    self.position += 1 # Compute self.position for the following operation.
                    self.parse() # Apply parse to these arguments.
                    if self.value() != ']': # Select this branch when its condition holds.
                        return node # Return this computed value to the caller.
                    end = tokens[self.position][2] # Compute end for the following operation.
                    self.position += 1 # Compute self.position for the following operation.
                    kind = node[2] # Compute kind for the following operation.
                    if kind and kind.endswith('[]'): # Select this branch when its condition holds.
                        kind = kind[:-2] # Compute kind for the following operation.
                    elif kind and 'vec' in kind: # Try this alternative condition after earlier branches fail.
                        kind = scalar(kind) # Compute kind for the following operation.
                    elif kind and kind.startswith('mat'): # Try this alternative condition after earlier branches fail.
                        kind = 'vec' + kind[-1] # Compute kind for the following operation.
                    else: # Handle the remaining case.
                        kind = None # Compute kind for the following operation.
                    node = (node[0], end, kind) # Compute node for the following operation.
                elif value == '.' and self.position + 1 < len(tokens): # Try this alternative condition after earlier branches fail.
                    field, _, end = tokens[self.position + 1] # Supply field for the continued expression.
                    self.position += 2 # Compute self.position for the following operation.
                    kind = node[2] # Compute kind for the following operation.
                    if field == 'length' and self.value() == '(' and self.position + 1 < len(tokens) and tokens[self.position + 1][0] == ')': # Every native GLSL empty array length query has signed int type.
                        end = tokens[self.position + 1][2] # Include both parentheses in the promoted method expression.
                        self.position += 2 # Consume the empty method argument list.
                        kind = 'int' # Apply signed GLSL length semantics before choosing numeric promotions.
                    elif kind in fields: # Ordinary structure members retain their declared component type.
                        kind = fields[kind].get(field) # Compute kind for the following operation.
                    elif ( # Try this alternative condition after earlier branches fail.
                        kind # Supply kind for the continued expression.
                        and 'vec' in kind # Supply and for the continued expression.
                        and re.fullmatch(r'[xyzwrgba stpq]+'.replace(' ', ''), field) # Apply fullmatch to these arguments.
                    ): # Finish this expression or collection.
                        kind = ( # Compute kind for the following operation.
                            scalar(kind) # Apply scalar to these arguments.
                            if len(field) == 1 # Select this branch when its condition holds.
                            else replace_scalar('vec' + str(len(field)), scalar(kind)) # Apply replace scalar to these arguments.
                        ) # Finish this expression or collection.
                    else: # Handle the remaining case.
                        kind = None # Compute kind for the following operation.
                    node = (node[0], end, kind) # Compute node for the following operation.
                elif value in {'++', '--'}: # Try this alternative condition after earlier branches fail.
                    node = (node[0], tokens[self.position][2], node[2]) # Compute node for the following operation.
                    self.position += 1 # Compute self.position for the following operation.
                else: # Handle the remaining case.
                    break # Finish this scan at the current boundary.
            return node # Return this computed value to the caller.

    for position, (value, _, _) in enumerate(tokens): # Process each position, (value,  ,  ) in this sequence.
        if (position == 0 or tokens[position - 1][0] != '.') and re.fullmatch( # Select this branch when its condition holds.
            r'[A-Za-z_]\w*|\d[\w.]*|\.[0-9][\w.]*', value # Supply r for the continued expression.
        ): # Finish this expression or collection.
            Expression(position).parse() # Apply Expression to these arguments.
    inserts = {} # Compute inserts for the following operation.
    for start, end, kind in conversions: # Process each start, end, kind in this sequence.
        inserts.setdefault(start, []).append((end, kind + '(')) # Apply setdefault to these arguments.
        inserts.setdefault(end, []).append((start, ')')) # Apply setdefault to these arguments.
    # Open nested casts widest-first and close them innermost-first to preserve evaluation order.
    for offset, items in sorted(inserts.items(), reverse=True): # Process each offset, items in this sequence.
        endings = ''.join( # Compute endings for the following operation.
            text for start, text in sorted(items, reverse=True) if text == ')' # Apply sorted to these arguments.
        ) # Finish this expression or collection.
        beginnings = ''.join( # Compute beginnings for the following operation.
            text for end, text in sorted(items, reverse=True) if text != ')' # Apply sorted to these arguments.
        ) # Finish this expression or collection.
        source = source[:offset] + endings + beginnings + source[offset:] # Compute source for the following operation.
    return source # Return this computed value to the caller.


def portable_integer_mix(source): # Supply boolean integer selection for every supported profile.
    """Supply integer boolean selection where older profiles lack that overload."""
    code = mask_noncode(source) # Compute code for the following operation.
    if not re.search(r'\bmix\s*\(', code) or re.search( # Select this branch when its condition holds.
        r'\b\w+\s+mix\s*\([^;{}]*\)\s*\{', code # Supply r for the continued expression.
    ): # Finish this expression or collection.
        return source # Return this computed value to the caller.
    name = 'shdr_portable_mix_' + hashlib.sha256(source.encode()).hexdigest()[:12] # Compute name for the following operation.
    helper = '#if (!defined(GL_ES) && __VERSION__ < 450) || (defined(GL_ES) && __VERSION__ < 320)\n' # Compute helper for the following operation.
    for scalar in ('float', 'int', 'uint'): # Process each scalar in this sequence.
        helper += ( # Compute helper for the following operation.
            scalar # Supply scalar for the continued expression.
            + ' '
            + name # Supply name for the continued expression.
            + '('
            + scalar # Supply scalar for the continued expression.
            + ' a,'
            + scalar # Supply scalar for the continued expression.
            + ' b,bool select_b){return select_b?b:a;}\n'
        ) # Finish this expression or collection.
        if scalar == 'float': # Select this branch when its condition holds.
            helper += ( # Compute helper for the following operation.
                'float '
                + name # Supply name for the continued expression.
                + '(float a,float b,float weight){return mix(a,b,weight);}\n'
            ) # Finish this expression or collection.
        for size in (2, 3, 4): # Process each size in this sequence.
            vector = ( # Compute vector for the following operation.
                'vec' if scalar == 'float' else 'ivec' if scalar == 'int' else 'uvec' # Supply vec for the continued expression.
            ) + str(size) # Apply str to these arguments.
            components = ','.join( # Compute components for the following operation.
                'select_b.' + c + '?b.' + c + ':a.' + c for c in 'xyzw'[:size] # Supply select b for the continued expression.
            ) # Finish this expression or collection.
            helper += ( # Compute helper for the following operation.
                vector # Supply vector for the continued expression.
                + ' '
                + name # Supply name for the continued expression.
                + '('
                + vector # Supply vector for the continued expression.
                + ' a,'
                + vector # Supply vector for the continued expression.
                + ' b,bvec'
                + str(size) # Apply str to these arguments.
                + ' select_b){return '
                + vector # Supply vector for the continued expression.
                + '('
                + components # Supply components for the continued expression.
                + ');}\n'
            ) # Finish this expression or collection.
            if scalar == 'float': # Select this branch when its condition holds.
                for weight in ('float', vector): # Process each weight in this sequence.
                    helper += ( # Compute helper for the following operation.
                        vector # Supply vector for the continued expression.
                        + ' '
                        + name # Supply name for the continued expression.
                        + '('
                        + vector # Supply vector for the continued expression.
                        + ' a,'
                        + vector # Supply vector for the continued expression.
                        + ' b,'
                        + weight # Supply weight for the continued expression.
                        + ' weight){return mix(a,b,weight);}\n'
                    ) # Finish this expression or collection.
    helper += '#define mix ' + name + '\n#endif\n' # Compute helper for the following operation.
    return helper + source # Return this computed value to the caller.


def needs_invocation_frame(source): # Detect private values whose native initialization must happen for every invocation.
    """Find initialized nonconst globals while ignoring interfaces and function locals."""
    code = mask_noncode(source) # Hide comments and quoted text before examining declaration scopes.
    for macro in re.finditer(r'(?m)^[ \t]*#[ \t]*define\b(?:[^\n]|(?<=\\)\n)*', code): # Include continuation lines when inspecting macro-defined private state.
        if re.search(r'(?<![=<>!])=(?!=)', macro.group()): # An assignment macro can introduce a mutable global after preprocessing.
            return True # Give macro-declared initialized globals a fresh invocation frame.
    code = re.sub(r'(?m)^[ \t]*#[^\n]*', lambda m: ' ' * len(m.group()), code) # Keep preprocessor text out of declaration matching.
    declarations = re.compile(r'\b(?P<qualifiers>(?:(?:const|highp|mediump|lowp)\s+)*)(?P<type>\w+)\s+\w+\s*(?:\[[^]]*\]\s*)*(?=[=,;])') # Accept scalar, vector, matrix, array and named private types.
    position, braces, parentheses = 0, 0, 0 # Track the scope preceding each possible global declaration.
    for match in declarations.finditer(code): # Inspect declarations in their original source order.
        for token in code[position:match.start()]: # Advance the lexical scope to this declaration's first token.
            if token == '{': # An opening brace enters a function, structure or initializer scope.
                braces += 1 # Keep this scan at the top-level declaration scope.
            elif token == '}': # A closing brace returns to the enclosing scope.
                braces -= 1 # Make subsequent top-level declarations eligible again.
            elif token == '(': # Distinguish parameters and call arguments from global declarations.
                parentheses += 1 # Keep this scan at the enclosing declaration level.
            elif token == ')': # A closing parenthesis ends that parameter or argument scope.
                parentheses -= 1 # Restore the enclosing declaration scope.
        position = match.start() # Continue counting from the same declaration boundary next time.
        if braces or parentheses or 'const' in match.group('qualifiers').split(): # Keep constants and local declarations in their existing scopes.
            continue # Keep the unchanged generated fast path for these declarations.
        prefix = code[:match.start()].rstrip() # Check qualifiers before a declaration matched from its numeric type.
        if re.search(r'\b(?:uniform|const)\s*$', prefix): # Uniform registration and constant values belong to the outer kernel.
            continue # Keep registered interfaces in their existing resource storage.
        end = statement_end(code, match.start()) # Include all declarators in a comma-separated global declaration.
        if '=' in code[match.end():end]: # Any initialized declarator in this statement needs fresh invocation state.
            return True # Select the frame only for shaders with initialized mutable globals.
    return False # Preserve existing adapter bytes for ordinary shader declarations.


def invocation_frame(cpu, name, interfaces): # Isolate native private globals from persistent kernel bindings.
    """Keep ports on the kernel and construct private shader state for each CPU call."""
    declarations, references, initializers = [], [], [] # Separate registered resources from fresh private implementation state.
    for builtin in ('gl_GlobalInvocationID', 'gl_LocalInvocationID', 'gl_WorkGroupID', 'gl_NumWorkGroups', 'gl_WorkGroupSize', 'gl_LocalInvocationIndex'): # Read the coordinates and sizes already prepared by the CPU dispatch loop.
        references.append('const decltype(::shdr::kernel::' + builtin + ')& ' + builtin + ';') # Keep the native builtin value immutable inside this invocation frame.
        initializers.append(builtin + '(owner.' + builtin + ')') # Bind the frame to this invocation's outer-kernel builtin fields before private initialization.
    seen = set() # A push-constant structure needs one reference even though it exposes several fields.
    for begin, end, variable, declaration, uniform in interfaces: # Extract only interface regions explicitly marked by transform.
        start = cpu.index(begin) # Locate the beginning of this generated interface declaration.
        stop = cpu.index(end, start) + len(end) # Include the end marker through its final character.
        declarations.append(cpu[start + len(begin):stop - len(end)]) # Retain the fully translated declaration on the outer kernel.
        cpu = cpu[:start] + '\n' * cpu[start:stop].count('\n') + cpu[stop:] # Leave private code and its line boundaries inside the frame.
        if variable not in seen: # Shared named blocks and push-constant fields reference one root object.
            qualifier = 'const ' if uniform else '' # Shader uniforms stay read-only inside invocation code.
            references.append(qualifier + 'decltype(' + name + '::' + variable + ')& ' + variable + ';') # Refer to the registered outer resource or uniform value.
            initializers.append(variable + '(owner.' + variable + ')') # Initialize references before any private member initializer runs.
            seen.add(variable) # Avoid declaring the same named interface reference twice.
    cpu = cpu.replace('SHDR_MAIN()', 'void main()') # Declare the frame entry point as an ordinary method.
    frame = 'shdr_invocation_' + hashlib.sha256(cpu.encode()).hexdigest()[:12] # Avoid colliding with shader-author names.
    initializer_list = ' : ' + ','.join(initializers) if initializers else '' # Give resource-free shaders fresh private state.
    return '\n'.join(declarations) + '\nstruct ' + frame + ' {\n' + '\n'.join(references) + '\nexplicit ' + frame + '(' + name + '& owner)' + initializer_list + ' {}\n' + cpu + '\n};\nvoid main() override { ' + frame + ' invocation(*this); invocation.main(); }\n' # Construct one private frame for every native CPU invocation.


def transform(source, name, interface_capture=None): # Produce the CPU body and embedded portable GPU body from native GLSL.
    """Return the C++ shader class and its GLSL text, interface forms are validated."""
    if not re.fullmatch(r'[A-Za-z_]\w*', name): # Select this branch when its condition holds.
        raise ShaderError('invalid shader name: ' + name) # Report the shader declaration validation error.
    def frame_interface(declaration, variable, uniform=False): # Record exact generated interface boundaries only when an invocation frame is required.
        if interface_capture is None: # Preserve the existing emitted text for ordinary shaders.
            return declaration # Preserve the fast-path adapter byte for byte.
        index = len(interface_capture) # Give each interface a unique generated marker pair.
        marker = 'shdr_frame_port_' + hashlib.sha256(source.encode()).hexdigest()[:12] + '_' + str(index) # Derive unique generated markers from this source.
        begin = '/* ' + marker + '_begin */' # Mark the start with a source comment.
        end = '/* ' + marker + '_end */' # Mark the matching end after any registered uniform descriptors.
        interface_capture.append((begin, end, variable, declaration, uniform)) # Keep the root name and access category for the frame reference.
        return begin + declaration + end # Let later ordinary CPU translations process the declaration unchanged.
    # shdr.h belongs to the CPU build only. All remaining shader input is self-contained.
    gpu = re.sub( # Compute gpu for the following operation.
        r'^[ \t]*#[ \t]*include[ \t]*[<"]shdr\.h[>"][ \t]*(?://[^\n]*|/\*[^\n]*?\*/[ \t]*)?$',
        '',
        source, # Supply source for the continued expression.
        flags=re.MULTILINE, # Compute flags for the following operation.
    ) # Finish this expression or collection.
    # The device selects its language profile; retain the imported shader body and line numbers.
    gpu = re.sub(r'^[ \t]*#[ \t]*version[^\n]*', '', gpu, flags=re.MULTILINE) # Compute gpu for the following operation.
    masked = mask_noncode(gpu) # Compute masked for the following operation.
    if re.search(r'^\s*#\s*include\b', masked, re.MULTILINE): # Select this branch when its condition holds.
        raise ShaderError( # Report the shader declaration validation error.
            'shader includes are unsupported; each shader is self-contained'
        ) # Finish this expression or collection.
    if re.search( # Select this branch when its condition holds.
        r'\b(?:shared|barrier|memoryBarrier\w*|groupMemoryBarrier|atomic\w*)\b', masked # Apply b to these arguments.
    ): # Finish this expression or collection.
        raise ShaderError( # Report the shader declaration validation error.
            'shared memory, barriers and atomics are outside the CPU portable subset'
        ) # Finish this expression or collection.
    replacements = [] # Compute replacements for the following operation.
    ports = [] # Compute ports for the following operation.
    named_arrays = set() # Compute named arrays for the following operation.
    covered = [] # Compute covered for the following operation.
    definitions = { # Compute definitions for the following operation.
        m.group(1): m.group(2).strip() # Apply group to these arguments.
        for m in re.finditer( # Process each m in this sequence.
            r'^[ \t]*#\s*define\s+(\w+)\s+([^\n]+)', masked, re.MULTILINE # Supply r for the continued expression.
        ) # Finish this expression or collection.
    } # Finish this expression or collection.
    constants = scoped_integer_constants(masked, definitions) # Compute constants for the following operation.
    resolve = lambda text, label, default=None: integer(text, label, default, constants) # Compute resolve for the following operation.
    local = [1, 1, 1] # Compute local for the following operation.
    local_matches = list(re.finditer(r'\blayout\s*\(([^)]*)\)\s*in\s*;', masked)) # Compute local matches for the following operation.
    gpu_edits = [] # Compute gpu edits for the following operation.
    seen_axes = set() # Compute seen axes for the following operation.
    for match in local_matches: # Process each match in this sequence.
        values = layout_values(match.group(1)) # Compute values for the following operation.
        if set(values) - { # Select this branch when its condition holds.
            'local_size_' + a + suffix for a in 'xyz' for suffix in ('', '_id') # Apply in to these arguments.
        }: # Finish this expression or collection.
            raise ShaderError('unsupported compute execution layout') # Report the shader declaration validation error.
        for i, axis in enumerate('xyz'): # Process each i, axis in this sequence.
            key = 'local_size_' + axis # Compute key for the following operation.
            if key not in values and key + '_id' not in values: # Select this branch when its condition holds.
                continue # Skip the remaining work for this iteration.
            value = resolve(values.get(key), key, 1) # Compute value for the following operation.
            if not value: # Select this branch when its condition holds.
                raise ShaderError('local workgroup dimensions must be positive') # Report the shader declaration validation error.
            if axis in seen_axes and local[i] != value: # Select this branch when its condition holds.
                raise ShaderError('conflicting local workgroup sizes') # Report the shader declaration validation error.
            seen_axes.add(axis) # Apply add to these arguments.
            local[i] = value # Supply local for the continued expression.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append((*match.span(), '')) # Apply append to these arguments.
    if local_matches: # Select this branch when its condition holds.
        for i, match in enumerate(local_matches): # Process each i, match in this sequence.
            gpu_edits.append( # Apply append to these arguments.
                ( # Finish this expression or collection.
                    *match.span(), # Apply span to these arguments.
                    ( # Finish this expression or collection.
                        ( # Finish this expression or collection.
                            'layout('
                            + ','.join( # Apply join to these arguments.
                                'local_size_' + a + '=' + str(local[j]) # Apply str to these arguments.
                                for j, a in enumerate('xyz') # Process each j, a in this sequence.
                            ) # Finish this expression or collection.
                            + ') in;'
                        ) # Finish this expression or collection.
                        if i == 0 # Select this branch when its condition holds.
                        else '' # Supply else for the continued expression.
                    ), # Finish this expression or collection.
                ) # Finish this expression or collection.
            ) # Finish this expression or collection.
    else: # Handle the remaining case.
        gpu_edits.append( # Apply append to these arguments.
            (0, 0, 'layout(local_size_x=1,local_size_y=1,local_size_z=1) in;\n') # Apply layout to these arguments.
        ) # Finish this expression or collection.

    # Vulkan specialization defaults become ordinary constants in an OpenGL import.
    for match in re.finditer( # Process each match in this sequence.
        r'\blayout\s*\(\s*constant_id\s*=\s*[^)]+\)\s*(const\s+\w+\s+\w+\s*=\s*[^;]+;)',
        masked, # Supply masked for the continued expression.
    ): # Finish this expression or collection.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append((*match.span(), match.group(1))) # Apply append to these arguments.
        gpu_edits.append((*match.span(), gpu[match.start(1) : match.end(1)])) # Apply append to these arguments.

    # A scalar/vector push-constant block maps to individually named uniforms.
    push_blocks = re.compile( # Compute push blocks for the following operation.
        r'\blayout\s*\(\s*push_constant\s*\)\s*uniform\s+(\w+)\s*\{([^{}]*)\}\s*(\w+)\s*;'
    ) # Finish this expression or collection.
    for match in push_blocks.finditer(masked): # Process each match in this sequence.
        block, body, instance = match.groups() # Apply groups to these arguments.
        fields = list(re.finditer(r'\b(\w+)\s+(\w+)\s*;', body)) # Compute fields for the following operation.
        remainder = body # Compute remainder for the following operation.
        for field in reversed(fields): # Process each field in this sequence.
            remainder = remainder[: field.start()] + remainder[field.end() :] # Compute remainder for the following operation.
        if ( # Select this branch when its condition holds.
            remainder.strip() # Apply strip to these arguments.
            or not fields # Supply or for the continued expression.
            or any(f.group(1) not in UNIFORMS for f in fields) # Apply any to these arguments.
        ): # Finish this expression or collection.
            raise ShaderError('push constants require scalar/vector/matrix fields') # Report the shader declaration validation error.
        declaration = ( # Compute declaration for the following operation.
            'struct '
            + block # Supply block for the continued expression.
            + ' { '
            + ' '.join(f.group() for f in fields) # Apply join to these arguments.
            + ' } '
            + instance # Supply instance for the continued expression.
            + '{};\n'
        ) # Finish this expression or collection.
        for field in fields: # Process each field in this sequence.
            scalar, variable = field.groups() # Apply groups to these arguments.
            port_name = instance + '.' + variable # Compute port name for the following operation.
            ports.append((port_name, 'uniform', 0)) # Apply append to these arguments.
            named_arrays.add((instance, variable)) # Apply add to these arguments.
            declaration += f'::shdr::uniform_port<{scalar}> shdr_uniform_{instance}_{variable}{{*this,"{port_name}",{port_name}}};\n'
        declaration = frame_interface(declaration, instance, True) # Keep the complete push-constant object registered outside private invocation state.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append((*match.span(), declaration)) # Apply append to these arguments.
        gpu_edits.append( # Apply append to these arguments.
            ( # Finish this expression or collection.
                *match.span(), # Apply span to these arguments.
                'struct '
                + block # Supply block for the continued expression.
                + ' {'
                + gpu[match.start(2) : match.end(2)] # Apply start to these arguments.
                + '}; uniform '
                + block # Supply block for the continued expression.
                + ' '
                + instance # Supply instance for the continued expression.
                + ';',
            ) # Finish this expression or collection.
        ) # Finish this expression or collection.

    default_buffer = False # Compute default buffer for the following operation.
    for match in re.finditer(r'\blayout\s*\(([^)]*)\)\s*buffer\s*;', masked): # Process each match in this sequence.
        values = layout_values(match.group(1)) # Compute values for the following operation.
        if 'std430' not in values or set(values) - {'std430', 'column_major'}: # Select this branch when its condition holds.
            raise ShaderError('default storage buffer layout must be std430') # Report the shader declaration validation error.
        default_buffer = True # Compute default buffer for the following operation.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append((*match.span(), '')) # Apply append to these arguments.
    buffers = re.compile( # Compute buffers for the following operation.
        r'(?:(?:\blayout\s*\(([^)]*)\)\s*))?('
        + QUALIFIERS # Supply QUALIFIERS for the continued expression.
        + r')\bbuffer\s+(\w+)\s*\{([^{}]*)\}\s*(\w+)?\s*;'
    ) # Finish this expression or collection.
    buffer_matches = list(buffers.finditer(masked)) # Compute buffer matches for the following operation.
    reserved_bindings = { # Compute reserved bindings for the following operation.
        resolve(layout_values(m.group(1) or '').get('binding'), 'SSBO binding') # Apply resolve to these arguments.
        for m in buffer_matches # Process each m in this sequence.
        if 'binding' in layout_values(m.group(1) or '') # Select this branch when its condition holds.
    } # Finish this expression or collection.
    for match in buffer_matches: # Process each match in this sequence.
        values = layout_values(match.group(1) or '') # Compute values for the following operation.
        if set(values) - {'std430', 'binding', 'column_major', 'set'}: # Select this branch when its condition holds.
            raise ShaderError('storage buffer layout must support std430') # Report the shader declaration validation error.
        if 'set' in values and resolve(values['set'], 'descriptor set') != 0: # Select this branch when its condition holds.
            raise ShaderError('OpenGL imports support only descriptor set zero') # Report the shader declaration validation error.
        if 'binding' in values: # Select this branch when its condition holds.
            binding = resolve(values['binding'], 'SSBO binding') # Compute binding for the following operation.
        else: # Handle the remaining case.
            binding = 0 # Compute binding for the following operation.
            while binding in reserved_bindings: # Continue while the parser still has matching input.
                binding += 1 # Compute binding for the following operation.
            reserved_bindings.add(binding) # Apply add to these arguments.
        qualifiers = 'std430,binding=' + str(binding) # Compute qualifiers for the following operation.
        if 'column_major' in values: # Select this branch when its condition holds.
            qualifiers += ',column_major' # Compute qualifiers for the following operation.
        if match.group(1) is not None: # Select this branch when its condition holds.
            close = masked.index(')', match.start()) + 1 # Compute close for the following operation.
            gpu_edits.append((match.start(), close, 'layout(' + qualifiers + ')')) # Apply append to these arguments.
        else: # Handle the remaining case.
            gpu_edits.append( # Apply append to these arguments.
                (match.start(), match.start(), 'layout(' + qualifiers + ') ') # Apply start to these arguments.
            ) # Finish this expression or collection.
        member = re.fullmatch( # Compute member for the following operation.
            r'\s*(?:(?:highp|mediump|lowp)\s+)?(\w+)\s+(\w+)\s*\[\s*([^]]*)\]\s*;\s*',
            match.group(4), # Apply group to these arguments.
        ) # Finish this expression or collection.
        if not member or member.group(1) not in STORAGE: # Select this branch when its condition holds.
            raise ShaderError( # Report the shader declaration validation error.
                'buffer blocks support one runtime-sized scalar/vec2/vec4 array'
            ) # Finish this expression or collection.
        scalar, variable, size = member.groups() # Apply groups to these arguments.
        fixed = resolve(size.strip(), 'buffer array length') if size.strip() else 0 # Compute fixed for the following operation.
        if size.strip() and not fixed: # Select this branch when its condition holds.
            raise ShaderError('fixed array length must be positive') # Report the shader declaration validation error.
        mode = access_mode(match.group(2)) # Compute mode for the following operation.
        if match.group(5): # Select this branch when its condition holds.
            instance = match.group(5) # Compute instance for the following operation.
            named_arrays.add((instance, variable)) # Apply add to these arguments.
            port_name = instance + '.' + variable # Compute port name for the following operation.
            declaration = ( # Compute declaration for the following operation.
                'struct shdr_block_' + match.group(3) + ' { ' # Apply group to these arguments.
                f'::shdr::buffer_port<{scalar},::shdr::access::{mode}> {variable}; '
                f'explicit shdr_block_{match.group(3)}(::shdr::kernel& p) : '
                f'{variable}(p,{binding},"{port_name}",{fixed}) {{}} }} {instance}{{*this}};'
            ) # Finish this expression or collection.
        else: # Handle the remaining case.
            port_name = variable # Compute port name for the following operation.
            declaration = f'::shdr::buffer_port<{scalar},::shdr::access::{mode}> {variable}{{*this,{binding},"{variable}",{fixed}}};'
        declaration = frame_interface(declaration, match.group(5) or variable) # Keep storage ports and their owning named block on the persistent kernel.
        ports.append((port_name, 'buffer', binding)) # Apply append to these arguments.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append((*match.span(), declaration)) # Apply append to these arguments.

    images = re.compile( # Compute images for the following operation.
        r'\blayout\s*\(([^)]*)\)\s*('
        + QUALIFIERS # Supply QUALIFIERS for the continued expression.
        + r')uniform\s+('
        + QUALIFIERS # Supply QUALIFIERS for the continued expression.
        + r')(image2D|iimage2D|uimage2D)\s+(\w+)\s*;'
    ) # Finish this expression or collection.
    for match in images.finditer(masked): # Process each match in this sequence.
        values = layout_values(match.group(1)) # Compute values for the following operation.
        formats = set(values) & set(IMAGES) # Compute formats for the following operation.
        if len(formats) != 1 or set(values) - formats - {'binding'}: # Select this branch when its condition holds.
            raise ShaderError( # Report the shader declaration validation error.
                'image requires binding=N and a supported r32/rgba32 format'
            ) # Finish this expression or collection.
        image_format = formats.pop() # Compute image format for the following operation.
        scalar, components, expected_type = IMAGES[image_format] # Supply scalar for the continued expression.
        if match.group(4) != expected_type: # Select this branch when its condition holds.
            raise ShaderError('image type and format do not match: ' + match.group(5)) # Report the shader declaration validation error.
        binding = resolve(values.get('binding'), 'image binding') # Compute binding for the following operation.
        mode = access_mode(match.group(2) + match.group(3)) # Compute mode for the following operation.
        if components == 4 and mode == 'readwrite': # Select this branch when its condition holds.
            raise ShaderError( # Report the shader declaration validation error.
                'portable RGBA images require separate readonly/writeonly declarations'
            ) # Finish this expression or collection.
        declaration = f'SHDR_IMAGE2D({binding},{image_format},{expected_type},{match.group(5)},{mode})'
        declaration = frame_interface(declaration, match.group(5)) # Keep the registered image port outside the fresh private frame.
        ports.append((match.group(5), 'image', binding)) # Apply append to these arguments.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append((*match.span(), declaration)) # Apply append to these arguments.

    uniforms = re.compile( # Compute uniforms for the following operation.
        r'(?:\blayout\s*\(([^)]*)\)\s*)?\buniform\s+(?:(?:highp|mediump|lowp)\s+)?(\w+)\s+(\w+)\s*;'
    ) # Finish this expression or collection.
    for match in uniforms.finditer(masked): # Process each match in this sequence.
        if any(a <= match.start() < b for a, b in covered): # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        if match.group(2) not in UNIFORMS: # Select this branch when its condition holds.
            raise ShaderError('unsupported uniform type: ' + match.group(2)) # Report the shader declaration validation error.
        if match.group(1) and set(layout_values(match.group(1))) - {'location'}: # Select this branch when its condition holds.
            raise ShaderError('unsupported uniform layout qualifier') # Report the shader declaration validation error.
        ports.append((match.group(3), 'uniform', 0)) # Apply append to these arguments.
        covered.append(match.span()) # Apply append to these arguments.
        replacements.append( # Apply append to these arguments.
            (*match.span(), frame_interface(f'SHDR_UNIFORM({match.group(2)},{match.group(3)})', match.group(3), True))
        ) # Finish this expression or collection.
    for match in re.finditer(r'\bprecision\s+(?:highp|mediump|lowp)\s+\w+\s*;', masked): # Process each match in this sequence.
        replacements.append((*match.span(), '')) # Apply append to these arguments.
    # GLSL array length queries return int even when the host resource API uses an unsigned size.
    for match in re.finditer(r'\.\s*length\s*\(\s*\)', masked): # Preserve native signed comparisons for both buffer ports and private arrays.
        if any(a <= match.start() < b for a, b in covered): # Replace resource declarations in their dedicated interface pass.
            continue # Leave declaration syntax to its existing interface conversion.
        first = primary_expression_start(masked, match.start()) # Locate the complete indexed, grouped or named-block receiver.
        replacements.append((first, first, 'static_cast<int>(')) # Open a signed cast around the existing swizzle and literal edits.
        replacements.append((match.end(), match.end(), ')')) # Close the cast after the complete zero-argument method call.
    # Convert native swizzles in the CPU adapter and retain GLSL swizzles in GPU code.
    for match in re.finditer(r'\.\s*([xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4})\b', masked): # Process each match in this sequence.
        pattern = match.group(1) # Compute pattern for the following operation.
        # Block array names may themselves be r, xy, rgba, etc. Only subsequent
        # vector member selections, such as state.xy[0].xy, are real swizzles.
        if (member_receiver(masked, match.start()), pattern) in named_arrays: # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        if len(pattern) == 1 and pattern in 'xyzw': # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        alphabet = ( # Compute alphabet for the following operation.
            'xyzw'
            if pattern[0] in 'xyzw' # Select this branch when its condition holds.
            else 'rgba' if pattern[0] in 'rgba' else 'stpq' # Supply else for the continued expression.
        ) # Finish this expression or collection.
        indices = ','.join(str(alphabet.index(c)) for c in pattern) # Compute indices for the following operation.
        replacements.append((*match.span(), '.template swizzle<' + indices + '>()')) # Apply append to these arguments.
    # Unsuffixed GLSL floating literals are float, whereas C++ would make them double.
    # Preserve scalar overload resolution and intermediate precision in the CPU adapter.
    floating = r'(?<![\w.])(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)[fF]?(?![\w.])' # Compute floating for the following operation.
    for match in re.finditer(floating, masked): # Process each match in this sequence.
        if match.group()[-1] not in 'fF': # Select this branch when its condition holds.
            replacements.append((*match.span(), match.group() + 'f')) # Apply append to these arguments.
    for match in re.finditer(r'\bnot\s*(?=\()', masked): # Process each match in this sequence.
        replacements.append((*match.span(), 'logicalNot')) # Apply append to these arguments.
    # Function parameter qualifiers have C++ reference equivalents.
    parameters = r'\b(inout|out|in)\s+(?:(?:highp|mediump|lowp)\s+)?(\w+)(?:\s*\[\s*([^]]+)\])?\s+(\w+)(?:\s*\[\s*([^]]+)\])?' # Compute parameters for the following operation.
    for match in re.finditer(parameters, masked): # Process each match in this sequence.
        if any(a <= match.start() < b for a, b in covered): # Select this branch when its condition holds.
            continue # Skip the remaining work for this iteration.
        qualifier, scalar, leading_size, variable, trailing_size = match.groups() # Apply groups to these arguments.
        if leading_size and trailing_size: # Select this branch when its condition holds.
            raise ShaderError('multidimensional helper arrays are unsupported') # Report the shader declaration validation error.
        size = leading_size or trailing_size # Compute size for the following operation.
        if size: # Select this branch when its condition holds.
            scalar = f'::shdr::glsl::array<{scalar},{resolve(size,"parameter array length")}>'
        replacements.append( # Apply append to these arguments.
            (*match.span(), scalar + (' ' if qualifier == 'in' else '& ') + variable) # Apply span to these arguments.
        ) # Finish this expression or collection.
    main = list(re.finditer(r'\bvoid\s+main\s*\(\s*(?:void\s*)?\)', masked)) # Compute main for the following operation.
    if len(main) != 1: # Select this branch when its condition holds.
        raise ShaderError('shader must define exactly one void main()') # Report the shader declaration validation error.
    replacements.append((*main[0].span(), 'SHDR_MAIN()')) # Apply append to these arguments.
    names = [port[0] for port in ports] # Compute names for the following operation.
    if len(names) != len(set(names)): # Select this branch when its condition holds.
        raise ShaderError('duplicate resource or uniform name') # Report the shader declaration validation error.
    for kind in ('buffer', 'image'): # Process each kind in this sequence.
        bindings = [port[2] for port in ports if port[1] == kind] # Compute bindings for the following operation.
        if len(bindings) != len(set(bindings)): # Select this branch when its condition holds.
            raise ShaderError('duplicate ' + kind + ' binding') # Report the shader declaration validation error.
    remaining = masked # Compute remaining for the following operation.
    for start, end in sorted(covered, reverse=True): # Process each start, end in this sequence.
        remaining = remaining[:start] + ' ' * (end - start) + remaining[end:] # Compute remaining for the following operation.
    if re.search(r'\b(?:layout|buffer|uniform)\b', remaining): # Select this branch when its condition holds.
        raise ShaderError( # Report the shader declaration validation error.
            'unsupported interface declaration; consult docs/portability.html'
        ) # Finish this expression or collection.
    cpu = gpu # Compute cpu for the following operation.
    previous = len(cpu) + 1 # Compute previous for the following operation.
    for start, end, replacement in sorted(replacements, reverse=True): # Process each start, end, replacement in this sequence.
        if end > previous: # Select this branch when its condition holds.
            raise ShaderError('overlapping shader declarations') # Report the shader declaration validation error.
        cpu = cpu[:start] + replacement + '\n' * gpu[start:end].count('\n') + cpu[end:] # Compute cpu for the following operation.
        previous = start # Compute previous for the following operation.
    # Use 32-bit CPU arithmetic for GLSL scalar precision declarations.
    cpu = re.sub(r'\b(?:highp|mediump|lowp)\b', '', cpu) # Compute cpu for the following operation.
    # Array types and constructors have direct C++11 value-wrapper equivalents.
    types = '|'.join(sorted(UNIFORMS, key=len, reverse=True)) # Compute types for the following operation.
    cpu = separate_array_declarators(cpu, types) # Compute cpu for the following operation.
    code = mask_noncode(cpu) # Compute code for the following operation.
    array_edits = [] # Compute array edits for the following operation.
    for match in re.finditer(r'\b(' + types + r')\s+(\w+)\s*\[\s*([^]]+)\s*\]', code): # Process each match in this sequence.
        count = integer( # Compute count for the following operation.
            match.group(3), # Apply group to these arguments.
            'private array length',
            constants=scoped_integer_constants(code, definitions, match.start()), # Compute constants for the following operation.
        ) # Finish this expression or collection.
        if not count: # Select this branch when its condition holds.
            raise ShaderError('private array length must be positive') # Report the shader declaration validation error.
        array_edits.append( # Apply append to these arguments.
            ( # Finish this expression or collection.
                *match.span(), # Apply span to these arguments.
                f'::shdr::glsl::array<{match.group(1)},{count}> {match.group(2)}',
            ) # Finish this expression or collection.
        ) # Finish this expression or collection.
    for match in re.finditer(r'\b(' + types + r')\s*\[\s*([^]]+)\s*\]', code): # Process each match in this sequence.
        count = integer( # Compute count for the following operation.
            match.group(2), # Apply group to these arguments.
            'array type length',
            constants=scoped_integer_constants(code, definitions, match.start()), # Compute constants for the following operation.
        ) # Finish this expression or collection.
        array_edits.append( # Apply append to these arguments.
            (*match.span(), f'::shdr::glsl::array<{match.group(1)},{count}>')
        ) # Finish this expression or collection.
    for match in re.finditer(r'\b(float|int|uint|bool)\s*(?=\()', code): # Process each match in this sequence.
        array_edits.append( # Apply append to these arguments.
            (*match.span(), f'::shdr::glsl::scalar_cast<{match.group(1)}>')
        ) # Finish this expression or collection.
    for start, end, replacement in sorted(array_edits, reverse=True): # Process each start, end, replacement in this sequence.
        cpu = cpu[:start] + replacement + cpu[end:] # Compute cpu for the following operation.
    # Keep directives meaningful to GCC's preprocessor; GLSL extension pragmas concern the GPU only.
    cpu = re.sub(r'^[ \t]*#\s*extension[^\n]*', '', cpu, flags=re.MULTILINE) # Compute cpu for the following operation.
    for start, end, replacement in sorted(gpu_edits, reverse=True): # Process each start, end, replacement in this sequence.
        gpu = gpu[:start] + replacement + gpu[end:] # Compute gpu for the following operation.
    gpu = portable_integer_mix(portable_gpu_promotions(portable_gpu_declarations(gpu))) # Compute gpu for the following operation.
    return cpu, gpu, local # Return this computed value to the caller.


def generate(source, name): # Emit the registered C++11 kernel and its embedded shader source.
    interfaces = [] if needs_invocation_frame(source) else None # Select fresh native private state only when the source needs it.
    cpu, gpu, local = transform(source, name, interfaces) # Translate the shader while marking frame interfaces when required.
    if interfaces is not None: # Give this interface-free shader its initialized private invocation state.
        cpu = invocation_frame(cpu, name, interfaces) # Recreate private globals for every CPU invocation after binding uniforms.
    delimiter = 'SHDR' + hashlib.sha256(gpu.encode()).hexdigest()[:10] # Compute delimiter for the following operation.
    return f'''// Generated from native GLSL .cpp; do not edit.
#include "shdr.h"
namespace {{
static const char shdr_embedded_source[] = R"{delimiter}({gpu}){delimiter}";
struct {name} : ::shdr::kernel {{
    {name}() : ::shdr::kernel("{name}",::shdr::extent({local[0]},{local[1]},{local[2]})) {{
        set_embedded_source(shdr_embedded_source);
    }}
    std::unique_ptr<::shdr::kernel> clone() const override {{
        std::unique_ptr<{name}> copy(new {name}); copy_interfaces(*copy);
        return std::unique_ptr<::shdr::kernel>(copy.release());
    }}
    using shader_uint = ::shdr::glsl::uint;
{cpu}
}};
::shdr::kernel* shdr_factory() {{ return new {name}; }}
::shdr::detail::shader_registration shdr_registration("{name}",shdr_factory);
}} // anonymous namespace
'''.replace(
        '    using shader_uint = ::shdr::glsl::uint;',
        '    // GLSL names are supplied by the header namespace.',
    ).replace( # Apply replace to these arguments.
        'namespace {\n', 'using namespace ::shdr::glsl;\nnamespace {\n', 1 # Supply namespace for the continued expression.
    ) # Finish this expression or collection.


def main(): # Read options and produce the requested generated adapter.
    parser = argparse.ArgumentParser(description=__doc__) # Compute parser for the following operation.
    parser.add_argument('--input', type=Path, required=True) # Apply add argument to these arguments.
    parser.add_argument('--output', type=Path, required=True) # Apply add argument to these arguments.
    parser.add_argument('--name', required=True) # Apply add argument to these arguments.
    args = parser.parse_args() # Compute args for the following operation.
    try: # Attempt the operation with explicit error handling below.
        output = generate(args.input.read_text(), args.name) # Compute output for the following operation.
    except (OSError, ShaderError) as error: # Convert this failure into the expected diagnostic.
        parser.exit(1, str(args.input) + ': ' + str(error) + '\n') # Apply exit to these arguments.
    args.output.parent.mkdir(parents=True, exist_ok=True) # Apply mkdir to these arguments.
    if not args.output.exists() or args.output.read_text() != output: # Avoid recompiling unchanged adapters when only the frontend implementation changed.
        args.output.write_text(output) # Update the generated file only when its complete emitted contents differ.


if __name__ == '__main__': # Select this branch when its condition holds.
    main() # Apply main to these arguments.
