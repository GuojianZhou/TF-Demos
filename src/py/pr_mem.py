#!/usr/bin/env python

def test_id():
    for i in range(-7,260):
        a=i
        print(id(a))
        b=i
        print("------ i : %d , id1 : 0x%x , id2 : 0x%x" % (i,id(a),id(b)))
        if id(a) != id(b):
            print("NNN i : %d , id1 : 0x%x , id2 : 0x%x" % (i,id(a),id(b)))
        else:
            print("=== i : %d , id : 0x%x" % (i,id(i)))

if __name__ == '__main__':
    test_id()
