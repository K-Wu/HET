from try_print_call_site import MyClass

if __name__ == "__main__":
    # ❯ python misc/playground/try_print_call_site_indirect.py
    # This should be logged by 'decorated_function' name_override decorated_function file_override try_print_call_site.py line_override 40
    # ------->|decorated_function() called:
    # ------->|    File "/home/kwu/HET/hrt/misc/playground/try_print_call_site_indirect.py", line 4, in <module>
    #     decorated_function()
    # I ran
    mc = MyClass()
    mc.decorated_function()
