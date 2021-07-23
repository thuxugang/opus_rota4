import numpy
import math, warnings

def calc_dihedral(c1, c2, c3, c4):
    
    v1 = Vector(c1[0], c1[1], c1[2])
    v2 = Vector(c2[0], c2[1], c2[2])
    v3 = Vector(c3[0], c3[1], c3[2])
    v4 = Vector(c4[0], c4[1], c4[2])
    
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = ab ** cb
    v = db ** cb
    w = u ** v
    angle = u.angle(v)
    try:
        if cb.angle(w) > 0.001:
            angle = -angle
    except ZeroDivisionError:
        pass
    return round(angle,2)

def calc_angle(c1, c2, c3): 
    v1 = Vector(c1[0], c1[1], c1[2])
    v2 = Vector(c2[0], c2[1], c2[2])
    v3 = Vector(c3[0], c3[1], c3[2])
    v1 = v1 - v2 
    v3 = v3 - v2 
    return v1.angle(v3) 


class Vector(object):
    "3D vector"

    def __init__(self, x, y=None, z=None):
        if y is None and z is None:
            if len(x) != 3:
                raise ValueError("Vector: x is not a "
                                "list/tuple/array of 3 numbers")
            self._ar = numpy.array(x, 'd')
        else:
            self._ar = numpy.array((x, y, z), 'd')

    def __repr__(self):
        x, y, z = self._ar
        return "<Vector %.2f, %.2f, %.2f>" % (x, y, z)

    def __neg__(self):
        "Return Vector(-x, -y, -z)"
        a = -self._ar
        return Vector(a)

    def __add__(self, other):
        "Return Vector+other Vector or scalar"
        if isinstance(other, Vector):
            a = self._ar + other._ar
        else:
            a = self._ar + numpy.array(other)
        return Vector(a)

    def __sub__(self, other):
        "Return Vector-other Vector or scalar"
        if isinstance(other, Vector):
            a = self._ar - other._ar
        else:
            a = self._ar - numpy.array(other)
        return Vector(a)

    def __mul__(self, other):
        "Return Vector.Vector (dot product)"
        return sum(self._ar * other._ar)

    def __div__(self, x):
        "Return Vector(coords/a)"
        a = self._ar / numpy.array(x)
        return Vector(a)

    def __pow__(self, other):
        "Return VectorxVector (cross product) or Vectorxscalar"
        if isinstance(other, Vector):
            a, b, c = self._ar
            d, e, f = other._ar
            c1 = numpy.linalg.det(numpy.array(((b, c), (e, f))))
            c2 = -numpy.linalg.det(numpy.array(((a, c), (d, f))))
            c3 = numpy.linalg.det(numpy.array(((a, b), (d, e))))
            return Vector(c1, c2, c3)
        else:
            a = self._ar * numpy.array(other)
            return Vector(a)

    def __getitem__(self, i):
        return self._ar[i]

    def __setitem__(self, i, value):
        self._ar[i] = value

    def __contains__(self, i):
        return (i in self._ar)

    def norm(self): 
        "Return vector norm" 
        return numpy.sqrt(sum(self._ar * self._ar)) 

    def left_multiply(self, matrix): 
        "Return Vector=Matrix x Vector" 
        a = numpy.dot(matrix, self._ar) 
        return Vector(a) 

    def angle(self, other):
        "Return angle between two vectors"
        n1 = self.norm()
        n2 = other.norm()
        c = (self * other) / (n1 * n2)
        c = min(c, 1)
        c = max(-1, c)
        return numpy.arccos(c)/numpy.pi*180
    def normalize(self): 
        "Normalize the Vector" 
        self._ar = self._ar / self.norm() 

    def copy(self): 
        "Return a deep copy of the Vector" 
        return Vector(self._ar) 
    def get_array(self): 
        "Return (a copy of) the array of coordinates" 
        return numpy.array(self._ar)         
        
def rotaxis(theta, vector): 

    vector = vector.copy() 
    vector.normalize() 
    c = numpy.cos(theta) 
    s = numpy.sin(theta) 
    t = 1 - c 
    x, y, z = vector.get_array() 
    rot = numpy.zeros((3, 3)) 
    # 1st row 
    rot[0, 0] = t * x * x + c 
    rot[0, 1] = t * x * y - s * z 
    rot[0, 2] = t * x * z + s * y 
    # 2nd row 
    rot[1, 0] = t * x * y + s * z 
    rot[1, 1] = t * y * y + c 
    rot[1, 2] = t * y * z - s * x 
     # 3rd row 
    rot[2, 0] = t * x * z - s * y 
    rot[2, 1] = t * y * z + s * x 
    rot[2, 2] = t * z * z + c 
    return rot 

def calculateCoordinates(refA, refB, refC, L, ang, di):
    
    AV=Vector(refA.position[0],refA.position[1],refA.position[2])
    BV=Vector(refB.position[0],refB.position[1],refB.position[2])
    CV=Vector(refC.position[0],refC.position[1],refC.position[2])
    
    CA=AV-CV
    CB=BV-CV

    ##CA vector
    AX=CA[0]
    AY=CA[1]
    AZ=CA[2]

    ##CB vector
    BX=CB[0]
    BY=CB[1]
    BZ=CB[2]

    ##Plane Parameters
    A=(AY*BZ)-(AZ*BY)
    B=(AZ*BX)-(AX*BZ)
    G=(AX*BY)-(AY*BX)

    ##Dot Product Constant
    F= math.sqrt(BX*BX + BY*BY + BZ*BZ) * L * math.cos(ang*(math.pi/180.0))

    ##Constants
    const=math.sqrt( math.pow((B*BZ-BY*G),2) *(-(F*F)*(A*A+B*B+G*G)+(B*B*(BX*BX+BZ*BZ) + A*A*(BY*BY+BZ*BZ)- (2*A*BX*BZ*G) + (BX*BX+ BY*BY)*G*G - (2*B*BY)*(A*BX+BZ*G))*L*L))
    denom= (B*B)*(BX*BX+BZ*BZ)+ (A*A)*(BY*BY+BZ*BZ) - (2*A*BX*BZ*G) + (BX*BX+BY*BY)*(G*G) - (2*B*BY)*(A*BX+BZ*G)

    X= ((B*B*BX*F)-(A*B*BY*F)+(F*G)*(-A*BZ+BX*G)+const)/denom

    if((B==0 or BZ==0) and (BY==0 or G==0)):
        const1=math.sqrt( G*G*(-A*A*X*X+(B*B+G*G)*(L-X)*(L+X)))
        Y= ((-A*B*X)+const1)/(B*B+G*G)
        Z= -(A*G*G*X+B*const1)/(G*(B*B+G*G))
    else:
        Y= ((A*A*BY*F)*(B*BZ-BY*G)+ G*( -F*math.pow(B*BZ-BY*G,2) + BX*const) - A*( B*B*BX*BZ*F- B*BX*BY*F*G + BZ*const)) / ((B*BZ-BY*G)*denom)
        Z= ((A*A*BZ*F)*(B*BZ-BY*G) + (B*F)*math.pow(B*BZ-BY*G,2) + (A*BX*F*G)*(-B*BZ+BY*G) - B*BX*const + A*BY*const) / ((B*BZ-BY*G)*denom)

    
    #GET THE NEW VECTOR from the orgin
    D=Vector(X, Y, Z) + CV
    with warnings.catch_warnings():
        # ignore inconsequential warning
        warnings.simplefilter("ignore")
        temp=calc_dihedral(AV, BV, CV, D)
    
  
    di=di-temp
    rot= rotaxis(math.pi*(di/180.0), CV-BV)
    D=(D-BV).left_multiply(rot)+BV
    
    return D.get_array()

