def need_translation():
    order = input('Do you need translation? (y/n): ')
    
    if order == 'y':
        return True
    elif order == 'n':
        return False
    else:
        print('Invalid input. Please enter "y" or "n".')
        return need_translation()