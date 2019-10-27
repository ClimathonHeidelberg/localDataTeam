#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(tidyverse)

load("./data.RData")
# Define UI for application that draws a histogram
ui <- fluidPage(
   
   # Application title
   titlePanel(h1("Klima zum Anfassen", align = "center",
                 style = "font-family: 'Chalkduster', bold;
                 font-weight: 600; line-height: 2; 
                 color: #4d3a7d;")),
   hr(style="border-color: #4d3a7d;"),
   plotOutput("co2History", height = 300),
   column(4, plotOutput("heatDays", height = 200)),
   column(8, plotOutput("recentCO2", height = 250),
          sliderInput("year", "Ziehen Sie den Knopf, um ein Jahr zu wählen!", value = 1960,min = 1960, 
                      max=2018, step=1, width =1000, sep=""),
          img(src='./qr-code-software.png', align = "right", width=100, height=100))
)

# Define server logic required to draw a histogram
server <- function(input, output) {
   
   output$co2History <- renderPlot({
     yearRange <- c(0, 200000, 400000, 600000, 800000)
     ggplot(co2Data, aes(x=year, y=CO2)) + geom_smooth(method = "loess", span=0.005, se=FALSE, col = "darkgreen", size =3) +
       scale_x_reverse(breaks = yearRange, labels = c("heute","vor\n200.000\nJahren","vor\n400.000\nJahren","vor\n 600.000\nJahren","vor\n800.000\nJahren")) +
       geom_rect(xmin = -6000, xmax = 1000, ymin = 160, ymax =390, col ="red", fill = NA, linetype = "dashed") +
       ylab("CO2-Konzentration") + xlab("") + ylim(150,400) + ggtitle("CO2-Konzentration in der Atmosphäre") +
       theme_bw() + theme(axis.text.x = element_text(size=20, face = "bold"), 
                          axis.text.y = element_text(size=15), 
                          axis.title = element_text(size=20),
                          plot.title = element_text(size=20, color = "darkred", hjust=0.5))
   })
   
   output$recentCO2 <- renderPlot({
     yint <- input$year
     ggplot(co2Today, aes(x=year, y=CO2)) + geom_line() + geom_point(col = "darkgreen", size =2) +
       ylab("CO2-Konzentration") + xlab("") + geom_vline(xintercept = yint, col = "red", linetype = "dashed", size=3) +
       scale_x_continuous(expand = c(0.1,0.1)) + ggtitle("CO2-Konzentration seit 1950") +
       theme_bw() + theme(axis.text.x = element_text(size=20, face = "bold"), 
                          axis.text.y = element_text(size=15), axis.title = element_text(size=20),
                          plot.title = element_text(size=20, hjust =0.5, face ="bold",, color = "salmon"))
   })
   
   output$heatDays <- renderPlot({
     i <- input$year
     colList <- colorRampPalette(c("green","yellow","red"))(length(unique(heatDay$days)))
     names(colList) <- sort(unique(heatDay$days))
     plotTab <- filter(heatDay, year == i) %>% 
       mutate(x=1, y=1, col = colList[as.character(days)])
     
     ggplot(plotTab, aes(x=x,y=y)) + geom_point(shape =21, fill = plotTab$col, alpha =0.8, size = plotTab$days+5) +
       geom_text(aes(label = days), face = "bold") + labs(title = "Tage extremer Hitze\n in Heidelberg") + 
       theme_void() + theme(plot.title = element_text(size = 20, hjust =0.5, face = "bold", color = "salmon"))
   })
}




# Run the application 
shinyApp(ui = ui, server = server)

