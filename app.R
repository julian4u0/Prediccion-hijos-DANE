# Load Shiny
library(shiny)
# Load ggplot2
library(ggplot2)

data <- read.csv("nuevos_datos.csv")

#use_python("/usr/local/bin/python")
#======================================================================================#
# Definir el UI para la
# Inicio UI
ui <- fluidPage(
    # titulos
    titlePanel(
        "Predicción de Hijos en Hogares Colombianos basado en datos del DANE"
    ),
    h4("Introduce las variables correspondientes:"),
    h5(
        a(
            href = "https://htmlpreview.github.io/?https://github.com/julian4u0/Prediccion-hijos-DANE/blob/main/Informe.html",
            "Informe del aplicativo"
        )
    ),
    
    # En side bar panel van todos los selectores
    sidebarLayout(
        sidebarPanel(
            #======================================================================================#
            #inicio selectores
            selectInput(
                "genero",
                "Genero del jefe del hogar:",
                c("Masculino" = 1,
                  "Femenino" = 2)
            ),
            sliderInput(
                "edad",
                "Edad del jefe del hogar:",
                min = 1,
                max = 100,
                value = 30
            ),
            sliderInput(
                "personas",
                "Cantidad de personas que conforman el hogar:",
                min = 1,
                max = 15,
                value = 3
            ),
            numericInput(
                "ingresos",
                "Ingresos del hogar:",
                min = 0,
                max = 100000000,
                value = 850000
            ),
            selectInput(
                "estadocivil",
                "Estado civil del jefe del hogar:",
                c(
                    "Casado(a)" = 6,
                    "Soltero(a)" = 5,
                    "Está separado(a) o divorciado(a)" = 4,
                    "Viudo(a)" = 3,
                    "No está casado(a) y vive en pareja hace dos años o más" = 2,
                    "No está casado(a) y vive en pareja hace menos de dos años" = 1
                )
            ),
            sliderInput(
                "cuartos",
                "¿En cuántos cuartos duermen las personas de este hogar?",
                min = 1,
                max = 10,
                value = 2
            ),
            selectInput(
                "etnia",
                "¿A cuál pueblo o etnia indígena pertenece el jefe del hogar?",
                c(
                    "Ninguna" = 6,
                    "Indígena" = 1,
                    "Gitano (a) (Rom)" = 2,
                    "Raizal del archipiélago de San Andrés, Providencia y Santa Catalina" = 3,
                    "Palenquero (a) de San Basilio" = 4,
                    "Negro (a), mulato (a) (afrodescendiente), afrocolombiano(a)" = 5
                )
            )
            
            
            #fin selectores
            #======================================================================================#
            
        ),
        
        #======================================================================================#
        # Mostrar output de server
        mainPanel(htmlOutput("textoPrediccion"), 
                  hr(),
                  h2("Grafico descriptivo de tu selección"),
                  hr(),
                  fluidRow(
                           column(6,plotOutput(outputId = "barplot_sexo", height = "300px")),
                           column(6,plotOutput(outputId = "hist_edad", height = "300px"))
                  ))
        
        # Fin mostrar output
        #======================================================================================#
    )
)
# fin UI

#======================================================================================#
# Define server logic

#inicio server
server <- function(input, output) {
    
    
    output$textoPrediccion <- renderText({
        paste(
            "<h4>Numero Estimado de hijos: ",
            modelo(input$genero, input$edad,input$estadocivil, 1, input$personas, input$ingresos, input$cuartos),
            "</h4>"
        )
    })
    
    output$barplot_sexo <- renderPlot({
        
        x<- c(1, 2)
        ggplot(data, aes(P6020)) +
            geom_bar(fill = ifelse(x == input$genero,'red','gray')) + 
            ylab("Frecuencia") + 
            xlab("Genero") + 
            ggtitle("Genero del jefe del hogar") + 
            scale_x_discrete(name = "Genero", limits = c("Masculino" , "Femenino")) 
        
        
    })
    
    output$hist_edad <- renderPlot({
        
        ggplot(data, aes(P6040)) +
            geom_histogram() + 
            ylab("Frecuencia") + 
            xlab("Genero") + 
            ggtitle("Edad del jefe del hogar") + 
            xlab("Edad (años)") +
            geom_vline(xintercept = input$edad, color = "red", linetype="dotted", size = 1.3) 
        
        
    })
}
#fin server

#======================================================================================#
# funciones
modelo <- function(genero, edad, estadocivil, etnia, personas, ingresos, cuartos) {
    #aca se puede llamar a un modelo
    resultado <-
        system(paste(
            c("python", "function_model_eval.py", genero, edad, estadocivil, "", personas, ingresos, cuartos),
            collapse = " "
        ),
        wait = TRUE,
        intern = T)
    
    return(resultado)
}
#fin funciones
#======================================================================================#

# Run the application
shinyApp(ui = ui, server = server)
#======================================================================================#