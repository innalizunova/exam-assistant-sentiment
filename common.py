def check_type(my_value, types_list, type_name):
    if types_list.count(my_value) == 0:
        raise ValueError(f'Unknown {type_name} type. Available types: {types_list}')
