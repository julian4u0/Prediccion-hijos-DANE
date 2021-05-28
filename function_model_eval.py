import sys
import neural_net

genero = sys.argv[1]
edad = sys.argv[2]
estadocivil = sys.argv[3]
etnia = sys.argv[4]
personas = sys.argv[5] # numero de personas que viven en el hogar
ingresos = sys.argv[6] #ingresos del hogar
cuartos = sys.argv[7] #cuartos


# 1 Sex of the chef of the house
# 2 Age of this chef
# 3 Marital status of this chef
# 4 Etnia
# 5 Number of persons living in the household
# 6 Income of the household
# 7 Number of rooms in which the persons of the household sleep



print(neural_net.label_new_data(int(genero), int(edad), int(estadocivil), etnia, int(personas), int(ingresos), int(cuartos)))
