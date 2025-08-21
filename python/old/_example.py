from roots import roots_z

f1 = lambda x, y, z: x-1
f2 = lambda x, y, z: y-1
f3 = lambda x, y, z: z
rts, R = roots_z(f1,f2,f3,[-1, -1, -1], [1,1,1], 3)
print(rts)