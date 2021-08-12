
####################################################################################################
# General Utility functions                                                                        #
####################################################################################################

def check_type(signature, **parameters):
    """Simple helper function that checks if the passed parameters of the function are of the specified types in ``signature``.

    General utility function.

    Args:
        signature (list):  
            A list of types containing the types of the function of which to check the passed parameters. For example for the function foo(a: str, b: bool, c: Union[int, float]) the signature list would look like [str, bool, (int, float)].

        parameters (dict):
            The parameters of the function as dict. Bascially the result of locals().
    
    Raises:
        TypeError: If the type of the parameter is not as specified.
        ValueError: If the number of passed signatures (``signature``) exceed the number of passed function parameters (``parameters``) or vice versa.
    """

    # the passed list containing type signatures must be as long as the passed parameters
    if len(signature) == len(parameters):
        for i, key in enumerate(parameters.keys()):
            # if a parameter can have multiple types it is specified as tuple in signatures
            if isinstance (signature[i], tuple):
                is_one_of_multiple_types = False

                # check if the parameter has either of the types defined in the tuple
                for multiple_type in signature[i]:
                    if isinstance(parameters[key], multiple_type):
                        is_one_of_multiple_types = True
                
                # otherwise raise a type error
                if not is_one_of_multiple_types:
                    raise TypeError(f'{key} has wrong type: got type \'{type(parameters[key]).__name__}\' but either of {" or ".join(repr(t.__name__) for t in signature[i])} expected.')

            # if the parameter can only have one type just check that one
            else:
                if not isinstance(parameters[key], signature[i]):
                    raise TypeError(f'paramter <{key}> has wrong type: got type \'{type(parameters[key]).__name__}\' but type \'{signature[i].__name__}\' expected.')

    # raise an error if the length is not equal
    else:
        if len(signature) > len(parameters):
            raise ValueError('There are more type signatures than paramaters, please make sure that the number of passed type signatures corresponds to the number of passed parameters.')
        else:
            raise ValueError('There are less type signatures than paramaters, please make sure that the number of passed type signatures corresponds to the number of passed parameters.')
            
        
def get_title_line(title):
    """Simple helper function that returns an formated title for writing into evaluation textfiles

    Utility function that can be used by a forecaster.

    Args:
        title (str):  
            The title to format of as string.
    
    Returns:
        A formatted string.
    """

    output = '-------------------------------------------------------------------------\n'
    output = output + str(title) + '\n'
    output = output + '-------------------------------------------------------------------------'

    return output


def write_to_report_file(lines_to_write, evaluation_file_path):
    """Simple helper function that writes one line or multiple lines to a file.

    General utility function.

    Args:
        lines_to_write (str or list):  
            The text to write to the file as a single line (str) or a list of lines (list of str).

        evaluation_file_path (Path):
            The path of the file to which to write the text as Path object.
    
    Raises:
        ValueError: If the ``lines_to_write`` are neither of type str nor of type list.
    """

    with open(evaluation_file_path, 'a+') as evaluation:
        append_at_end = False
        
        if evaluation_file_path.stat().st_size != 0:
            append_at_end = True

        if isinstance(lines_to_write, list):
            for line in lines_to_write:
                if append_at_end:
                    evaluation.write('\n')
                
                evaluation.write(line)
        
        elif isinstance(lines_to_write, str):
            if append_at_end:
                evaluation.write('\n')
                
            evaluation.write(lines_to_write)

        else:
            raise ValueError('lines_to_write must be either a list of strings or a string.')
