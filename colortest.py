#!/usr/bin/python

#    Set ANSI Terminal Color and Attributes.
# Original code obtained from: http://code.activestate.com/recipes/574451/
#
from __future__ import (absolute_import, division, print_function)
from sys import stdout

esc = '%s[' % chr(27)
reset = '%s0m' % esc
format_code = '1;%dm'
fgoffset, bgoffset = 30, 40

for k, v in dict(attrs='none bold faint italic underline blink fast reverse concealed',
                 colors='grey red green yellow blue magenta cyan white').items():
    globals()[k] = dict((s, i) for i, s in enumerate(v.split()))


def term(arg=None, sep=' ', end='\n'):
    """
        "arg" is a string or None
        if "arg" is None : the terminal is reset to its default values.
        if "arg" is a string it must contain "sep" separated values.

        if args are found in globals "attrs" or "colors", or start with "@"
        they are interpreted as ANSI commands else they are output as text.

        colors, if any, must be first (foreground first then background)
        you can not specify a background color alone ; \
    if you specify only one color, it will be the foreground one.
        @* commands handle the screen and the cursor :
            @x;y : go to xy
            @    : go to 1;1
            @@   : clear screen and go to 1;1

        examples:
    term('red')                  : set red as the foreground color
    term('red blue')             : red on blue
    term('red blink')            : blinking red
    term()                       : restore terminal default values
    term('reverse')              : swap default colors
    term('cyan blue reverse')    : blue on cyan <=> term('blue cyan)
    term('red reverse')          : a way to set up the background only
    term('red reverse blink')    : you can specify any combination of \
            attributes in any order with or without colors
    term('blink Python')         : output a blinking 'Python'
    term('@@ hello')             : clear the screen and print 'hello' at 1;1
    """
    cmd, txt = [reset], []

    if arg:
        arglist = arg.split(sep)

        for offset in (fgoffset, bgoffset):
            if arglist and arglist[0] in colors:
                cmd.append(format_code % (colors[arglist.pop(0)] + offset))

        for a in arglist:
            c = format_code % attrs[a] if a in attrs else None
            if c and c not in cmd:
                cmd.append(c)
            else:
                if a.startswith('@'):
                    a = a[1:]
                    if a == '@':
                        cmd.append('2J')
                        cmd.append('H')
                    else:
                        cmd.append('%sH' % a)
                else:
                    txt.append(a)
    if txt and end:
        txt[-1] += end

    stdout.write(esc.join(cmd) + sep.join(txt))


if __name__ == '__main__':
    from time import sleep, strftime, gmtime

    print('')
    term('@@ reverse blink')
    print('reverse blink  default colors at 1;1 on a cleared screen')
    term('red')
    print('red')
    term('red blue')
    print('red blue')
    term('yellow blink')
    print('yellow blink')
    term('default')
    term('cyan blue cyan blue')
    term('cyan blue reverse cyan blue reverse')
    term('blue cyan blue cyan')
    term('red reverse red reverse')
    term('yellow red yellow on red 1')
    term('yellow,red,yellow on red 2', sep=',')
    print('yellow on red 3')
    print('')

    for bg in colors:
        term(bg.title().center(8), sep='.', end='')

        for fg in colors:
            att = [fg, bg]

            if fg == bg:
                att.append('blink')

            att.append(fg.center(8))
            term(','.join(att), sep=',', end='')
        print('')
    print('')

    for att in attrs:
        term('%s,%s' % (att, att.title().center(10)), sep=',', end='')
    print('')

    colist = 'grey blue cyan white cyan blue'.split()

    while True:
        try:
            for cc in colist:
                sleep(.1)
                term('%s @28;33 hit ctrl-c to quit' % cc)
            term('yellow @6;66 %s' % strftime('%H:%M:%S', gmtime()))
        except KeyboardInterrupt:
            break
        except:
            raise
    term('@30;1')
    print('')

