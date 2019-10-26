#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

load("./data.RData")
# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel("CO2 level over time"),
   plotOutput("co2History"),
   column(4, plotOutput("heatDays")),
   column(8, plotOutput("recentCO2", height = 300),
          sliderInput("yearChoice", "Slide to choose the year", value = 1960,min = 1960, 
                      max=2018, step=1, width =1000, sep=""))
)

# Define server logic required to draw a histogram
server <- function(input, output) {
   
   output$co2History <- renderPlot({
     yearRange <- c(0, 200000, 400000, 600000, 800000)
     ggplot(co2Data, aes(x=year, y=CO2)) + geom_smooth(method = "loess", span=0.005, se=FALSE) +
       scale_x_reverse(breaks = yearRange, labels = c("heute","vor\n200.000\nJahren","vor\n400.000\nJahren","vor\n 600.000\nJahren","vor\n800.000\nJahren")) +
       geom_rect(xmin = -6000, xmax = 1000, ymin = 160, ymax =390, col ="red", fill = NA, linetype = "dashed") +
       ylab("CO2-Konzentration") + xlab("") + ylim(150,400) +
       theme_bw() + theme(axis.text.x = element_text(size=20, face = "bold"), 
                          axis.text.y = element_text(size=15), axis.title = element_text(size=20))
   })
   
   output$recentCO2 <- renderPlot({
     yint <- input$yearChoice
     ggplot(co2Today, aes(x=year, y=CO2)) + geom_line() + geom_point() +
       ylab("CO2-Konzentration") + xlab("") + geom_vline(xintercept = yint, col = "red", linetype = "dashed") +
       scale_x_continuous(expand = c(0.1,0.1)) +
       theme_bw() + theme(axis.text.x = element_text(size=20, face = "bold"), 
                          axis.text.y = element_text(size=15), axis.title = element_text(size=20))
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

