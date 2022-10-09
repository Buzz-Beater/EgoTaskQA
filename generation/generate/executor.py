"""

    Created on 2022/5/17

    @author: Baoxiong Jia

"""

import copy as cp
import generate.operations as gu


def execute(program, intervals_data, causal_trace):
    """
    :param program:         A list of operations with arguments
    :param intervals_data:  the result
    :return:
    """
    stack = []
    result_stack = []
    current_string = ""
    arguments = []
    program = list(cp.deepcopy(program))
    ori_program = cp.deepcopy("".join(program))
    within_noun = False
    while len(program) != 0:
        ch = program.pop(0)
        if ch == " " or ch == "\t":
            if within_noun:
                current_string += ch
            continue
        if ch == "'":
            if not within_noun:
                within_noun = True
            else:
                within_noun = False
        elif ch == ",":
            if within_noun:
                current_string += ch
            if current_string != "":
                stack.append(current_string)
                current_string = ""
        elif ch == "[":
            if within_noun:
                current_string += ch
                continue
            if current_string != "":
                stack.append(current_string)
                current_string = ""
            stack.append(ch)
        elif ch == "]":
            if within_noun:
                current_string += ch
                continue
            if current_string != "":
                stack.append(current_string)
                current_string = ""
            arg = stack.pop()
            while arg != "[":
                arguments.insert(0, arg)
                arg = stack.pop()
            stack.append(arguments)
            arguments = []
        elif ch == "<":
            if within_noun:
                current_string += ch
                continue
            if current_string != "":
                stack.append(current_string)
                current_string = ""
            stack.append(ch)
        elif ch == ">":
            if within_noun:
                current_string += ch
                continue
            if current_string != "":
                stack.append(current_string)
                current_string = ""
            # Remove useless stack containments
            key = result_stack.pop()
            stack.pop()
            stack.pop()             # pop "<"
            arg = stack.pop() + key
            stack.append(arg)
        elif ch == "(":
            if within_noun:
                current_string += ch
                continue
            if current_string != "":
                stack.append(current_string)
                current_string = ""
            stack.append(ch)
        elif ch in ")":
            if within_noun:
                current_string += ch
                continue
            if current_string != "":
                stack.append(current_string)
                current_string = ""
            op_arguments = []
            arg = stack.pop()
            while arg != "(":
                op_arguments.insert(0, arg)
                arg = stack.pop()
            operation = stack.pop()
            if len(op_arguments) > 0 and op_arguments[-1] in ["video", "all"]:
                op_arguments += [intervals_data, causal_trace]
            else:
                if operation not in ["depend", "localize"]:
                    interval = result_stack.pop()
                    op_arguments += [interval, intervals_data, causal_trace]
                else:
                    stack_arguments = [result_stack.pop()]
                    stack_arguments.insert(0, result_stack.pop())
                    op_arguments += stack_arguments
                    op_arguments += [intervals_data, causal_trace]
            results = getattr(gu, operation)(*op_arguments)
            if results is None:
                return None
            result_stack.append(results)
            if operation == "query":
                stack.append(results)
        else:
            current_string += ch
    assert len(result_stack) == 1, "After computation the overall results should be one single output"
    return result_stack[0]

