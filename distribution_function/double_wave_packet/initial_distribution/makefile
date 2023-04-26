FC      = mpiifort
FCFLAGS = -O4 -parallel
LDFLAGS = -qopenmp
TARGET  = a.out

OBJ  = mt19937.o \
       constant_parameter.o \
       lshell_setting.o \
       von_Neumann.o \

all: $(TARGET)

.SUFFIXES : .f90

.f90.o:
	$(FC) -c $< $(FCFLAGS) $(LDFLAGS)

$(TARGET): $(OBJ)
	$(FC) $(OBJ) $(FCFLAGS) $(LDFLAGS)

clean:
	rm -f *.o *.mod *.exe core