# char-rnn-names
Character Level RNNs for Modeling Names

### Embeddings
![Embeddings](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/embedding.png)
All the _vowels_ are together.

### Name Generator Setup
![model](https://raw.githubusercontent.com/madaan/char-rnn-names/master/docs/model.png)

| names              | seeds         | african_american           | caucasian                  | hispanic                   | indian                     | all_races                |
|--------------------|---------------|----------------------------|----------------------------|----------------------------|----------------------------|--------------------------|
| derek vroom        | de            | derrick l curtis#          | dennis r coleman#          | dennis r suro#             | deepak . pardeep#          | derrick l brown#         |
| derek vroom        | derek         | derek a clark#             | derek a hardellin#         | derek m sanchez#           | derekha  bhai#             | derek a carter#          |
| derek vroom        | derek vr      | derek vright#              | derek vraworit#            | derek vrkilla.carcial#     | derek vrasha#              | derek vrow#              |
| dan fineman        | da            | david l morgan#            | david l mccoy#             | daniel carachure#          | dalip shah#                | david l brown#           |
| dan fineman        | dan f         | dan f briggs#              | dan f witharr#             | dan flekrez#               | dan farjat saini#          | dan francersiii#         |
| dan fineman        | dan fine      | dan finette tyrel#         | dan fines#                 | dan finera.garcia#         | dan fine#                  | dan fine#                |
| carolyn kennedy    | car           | carlos e jr robinson#      | carlos r reyes#            | carlos a guerra#           | carta#                     | carlos a cardona#        |
| carolyn kennedy    | carolyn       | carolyn r miller#          | carolyn a roe#             | carolyno  jr rodriguez#    | carolynir#                 | carolyn r martinez#      |
| carolyn kennedy    | carolyn ken   | carolyn kendry#            | carolyn kendrick#          | carolyn kendeles.rodriguez | carolyn kent#              | carolyn kent#            |
| michael zuckerberg | mich          | michael a chapman#         | michael a mckinney#        | michael a martinez#        | michael#                   | michael a mccormick#     |
| michael zuckerberg | michael z     | michael z chatman#         | michael z baker#           | michael z martinez#        | michael zah#               | michael z russell#       |
| michael zuckerberg | michael zucke | michael zuckenne#          | michael zuckellan#         | michael zucke#             | michael zuckena#           | michael zuckelli#        |
| gary yao           | ga            | gary l hayes#              | gary w jr davis#           | gabriel a romero#          | gaurav sharam#             | gary l baker#            |
| gary yao           | gary          | gary l hayes#              | gary w jr davis#           | gary avila#                | gary#                      | gary l baker#            |
| gary yao           | gary y        | gary y jr brown#           | gary yancey#               | gary y avares#             | gary yadav#                | gary yadav#              |
| pranay chopra      | pra           | pradel lucas#              | praie r saliska#           | pramie rivera#             | pramod kumar tiwari#       | pramod kumar tiwari#     |
| pranay chopra      | pranay        | pranay s mccarter#         | pranayoos n santana#       | pranay a gonzalez#         | pranayavara bijgut#        | pranayan devi#           |
| pranay chopra      | pranay ch     | pranay charlon#            | pranay chess#              | pranay chaveda#            | pranay chawla#             | pranay chand#            |
| mayank bhardwaj    | may           | maydreona l miles#         | maykel c calderon#         | maynor padilla.estrada#    | maya kumari#               | maya devi#               |
| mayank bhardwaj    | mayank        | mayank redison#            | mayank j romero#           | mayank h rodriguez#        | mayank .aakash#            | mayank pandey#           |
| mayank bhardwaj    | mayank bhar   | mayank bharder#            | mayank bharresse#          | mayank bharque#            | mayank bhardwaj#           | mayank bhardwaj#         |
| aman madaan        | am            | amos e robinson#           | amanda l hatcher#          | amalio rivera#             | amit kumar singh#          | amanda l hutchinson#     |
| aman madaan        | aman          | aman j joseph#             | aman p jr marshall#        | aman f rivera#             | aman saroj#                | aman sharma#             |
| aman madaan        | aman mad      | aman madadenis#            | aman madich#               | aman madro l gonzalez#     | aman madhkaran#            | aman madha#              |
| priyanka lingawal  | priy          | priy loveranc#             | priye j mergtone#          | priy g martinez#           | priyanka . pinki#          | priyanka . pinki#        |
| priyanka lingawal  | priyanka      | priyankauguson#            | priyanka l monagan#        | priyankaune#               | priyanka . pinki#          | priyanka . pinki#        |
| priyanka lingawal  | priyanka lin  | priyanka linkley#          | priyanka linto#            | priyanka linorriz#         | priyanka linditay#         | priyanka linda#          |
| uma sawant         | um            | umar a hassan#             | umid r mann#               | umeliar m sanchez#         | uma . barkha#              | uma devi#                |
| uma sawant         | uma s         | uma stimer l jenkins#      | uma s vancey#              | uma s muniz#               | uma shankar soni#          | uma shankar#             |
| uma sawant         | uma saw       | uma sawrm crow#            | uma sawyornig#             | uma sawillas#              | uma sawan#                 | uma saweer#              |
| ankur dewan        | an            | anthony l mcclain#         | anthony r horn#            | antonio r hernandez#       | anil kumar sain#           | anthony j perry#         |
| ankur dewan        | ankur         | ankurick reed#             | ankur m davis#             | ankur mondales#            | ankur kumar#               | ankur sharma#            |
| ankur dewan        | ankur de      | ankur dewin#               | ankur dehalf#              | ankur delapamill#          | ankur devi#                | ankur devi#              |
| rahul gupta        | ra            | raymond l hopps#           | randall l rice#            | ramon a feliciano#         | ram sagar#                 | raymond l holland#       |
| rahul gupta        | rahul         | rahuleshor c robbin#       | rahul a jr gonzalez#       | rahul g martinez#          | rahul sharma#              | rahul sharma#            |
| rahul gupta        | rahul gu      | rahul gudert#              | rahul guenoa#              | rahul gutineza#            | rahul gupta#               | rahul gupta#             |
| zhang wei          | zh            | zhivago rodgers#           | zhenya v deckarri#         | zhecory colinza#           | zhini devi#                | zhenya santana#          |
| zhang wei          | zhan          | zhankhea l stencor#        | zhane a nelson#            | zhanole s estrada#         | zhanna sankar#             | zhanson e martin#        |
| zhang wei          | zhang         | zhang m forbeson#          | zhang j henderson#         | zhang g diaz.hernandez#    | zhang depo#                | zhang r johnson#         |
| wang xiu ying      | wan           | wanda j malone#            | wanda j dentoscionle#      | wanda rodriguez#           | wani bhavi#                | wanda j davis#           |
| wang xiu ying      | wang x        | wang x jamison#            | wang x bailey#             | wang x rodriguez#          | wang xai d.o lageh ri par# | wang xiluseiia mond#     |
| wang xiu ying      | wang xiu      | wang xiu douglas#          | wang xiu r mccall#         | wang xiu castillo#         | wang xiu shaker#           | wang xiu santing#        |
| hideyoshi          | hi            | hillard j london#          | hilario m gomez#           | hiram a maldonado#         | himanshu . annu#           | himanshu . annu#         |
| hideyoshi          | hide          | hideajrake redmon#         | hidel f martinez#          | hidel garcia#              | hidee chaudhry#            | hidena#                  |
| hideyoshi          | hideyo        | hideyon u bennett#         | hideyo g morio#            | hideyordo rodriguez#       | hideyohar sharma#          | hideyon d brown#         |
| zhang xiu ying     | zha           | zharran e austin#          | zhays j notz#              | zhanole s estrada#         | zhanna sankar#             | zhayni essidion#         |
| zhang xiu ying     | zhang x       | zhang x hall#              | zhang x herty#             | zhang x corrova#           | zhang xeta#                | zhang x i barraman#      |
| zhang xiu ying     | zhang xiu     | zhang xiu donner#          | zhang xiu r brack#         | zhang xiu cardenas#        | zhang xiu w.o#             | zhang xiu partadill#     |
| hashimoto          | ha            | harry l jr caine#          | harold d minor#            | harold a cardenas.valenci# | harish chand dua#          | harold l harris#         |
| hashimoto          | hash          | hashantae walleggter#      | hashia l sannero.ferzill.c | hashus rodriguez.gutierre# | hashim#                    | hashim#                  |
| hashimoto          | hashim        | hashimmela j slove#        | hashim naviguenteras#      | hashimaer m gonzalez.matha | hashim#                    | hashim#                  |
| shinzo abe         | sh            | shantae d dennis#          | shannon l bennett#         | shawn d rodriguez#         | shankar lal gopalani#      | shane a brown#           |
| shinzo abe         | shinz         | shinzella b dowers#        | shinzeyne a scharden#      | shinz a medina#            | shinz yadav#               | shinza delvacy#          |
| shinzo abe         | shinzo        | shinzo  iii hardy#         | shinzo a dejesunnell.igos# | shinzo d rodriguez#        | shinzo . raju#             | shinzo . ammo#           |
| maria guadalupe    | mar           | marcus d cook#             | mark a beasley#            | mario a martinez#          | mariyam . sonam#           | mark a barnes#           |
| maria guadalupe    | maria g       | maria green#               | maria g pierce#            | maria g serrano#           | maria garg#                | maria g castillo#        |
| maria guadalupe    | maria guada   | maria guada#               | maria guadaguez#           | maria guadal.pont#         | maria guadal#              | maria guada#             |
| jose luis          | jo            | john l graham#             | joseph a bernston#         | jose a correa#             | joyti sharma#              | joseph a brown#          |
| jose luis          | jose          | joseph l henry#            | joseph a bernston#         | jose a correa#             | josender  sharma#          | joseph a brown#          |
| jose luis          | jose l        | jose l graham#             | jose l ramirez#            | jose l morales#            | jose lal sharma#           | jose l rodriguez#        |
| veronica           | ve            | vernon l hankerson#        | vernon r locke#            | veronica hernandez#        | veer bhan#                 | vernon l brown#          |
| veronica           | vero          | veronica m butler#         | veronica b jamison#        | veronica hernandez#        | veronika bhinder#          | veronica l brown#        |
| veronica           | veroni        | veronica m butler#         | veronica b jamison#        | veronica hernandez#        | veronika bhinder#          | veronica l brown#        |
| juan carlos        | ju            | julius mcdaniels#          | justin a moore#            | juan c montalba#           | julfi . sonu#              | justin l barnette#       |
| juan carlos        | juan          | juan p laster#             | juan c martinez#           | juan c montalba#           | juan morji#                | juan c castillo#         |
| juan carlos        | juan car      | juan cardonas#             | juan carranza#             | juan carrillo#             | juan carodhari#            | juan cardona#            |
| rosa maria         | ro            | robert l moore#            | robert l bresteane#        | roberto c munoz rivera#    | rohit agrawal#             | robert l jr marshall#    |
| rosa maria         | rosa          | rosa a hobbs#              | rosa m demerd#             | rosa m torres#             | rosa kalah#                | rosa a johnson#          |
| rosa maria         | rosa ma       | rosa mannt#                | rosa martinez#             | rosa mata#                 | rosa mani#                 | rosa massey#             |
| francisco javier   | fran          | frank j brown#             | frank e carballo#          | francisco j mendez#        | frana#                     | frank l jr bennett#      |
| francisco javier   | francisc      | francisco a nunez#         | francisco gonzales#        | francisco j mendez#        | francischan#               | francisco j mendez#      |
| francisco javier   | francisco ja  | francisco javen#           | francisco jaza.torrez#     | francisco jaigue#          | francisco jain#            | francisco jaramaz.ramos# |
| maria elena        | ma            | marcus d cook#             | mark a beasley#            | mario a martinez#          | manish kumar  mandal#      | mark a barnes#           |
| maria elena        | maria         | maria murray#              | maria l west#              | maria g serrano#           | mariam  mumtaz#            | maria l lassinorino#     |
| maria elena        | maria el      | maria ellis#               | maria elweow#              | maria elaramdo#            | maria elam#                | maria eller#             |
| brianna            | b             | brandon m fleming#         | brandon l holland#         | bryan santana#             | bharat bhushan#            | brian k bradley#         |
| brianna            | bri           | brian k brickhound#        | brian k saitham#           | brian n acosta#            | brij mohan thakur#         | brian k bradley#         |
| brianna            | brian         | brian k brickhound#        | brian k saitham#           | brian n acosta#            | brianti#                   | brian k bradley#         |
| chloe              | c             | charles e hall#            | christopher m brannon#     | carlos a guerra#           | chander shekhar#           | christopher m mcclendon# |
| chloe              | ch            | charles e hall#            | christopher m brannon#     | christian rivera#          | chander shekhar#           | christopher m mcclendon# |
| chloe              | chl           | chlente chasbin#           | chloe p johnson#           | chlistonae a funsez#       | chlepal#                   | chletu s jr marshall#    |
| destiny            | d             | david l morgan#            | david l mccoy#             | daniel carachure#          | deepak . pardeep#          | david l brown#           |
| destiny            | des           | desmond t baker#           | desiree l barnett#         | dessie santiago#           | desh raj#                  | desmond t harris#        |
| destiny            | desti         | destiny j rios#            | destiny straton#           | destin santiago#           | desti#                     | destiny j brown#         |
| jeremiah           | je            | jermaine l martin#         | jeremy m morrison#         | jesus m salinas#           | jeet pal  .jitu#           | jeremy l coleman#        |
| jeremiah           | jere          | jeremy a lee#              | jeremy m morrison#         | jeremy o burgos#           | jerender singh#            | jeremy l coleman#        |
| jeremiah           | jeremi        | jeremiah d preister#       | jeremiah m conner#         | jeremias v gomez#          | jeremi  rana#              | jeremiah d brown#        |
| josiah             | j             | james l caldwell#          | joseph a bernston#         | jose a correa#             | jai kishan gupta#          | joseph a brown#          |
| josiah             | jos           | joseph l henry#            | joseph a bernston#         | jose a correa#             | josin jomon#               | joseph a brown#          |
| josiah             | josi          | josiah m beachem#          | josian montanez#           | josie a rivera#            | josin jomon#               | josiah m martinez#       |
| undertaker         | un            | undray anderson#           | uney s mendes#             | unesto brito#              | unknown . monu#            | undrae m mcclellan#      |
| undertaker         | under         | underially r ivy#          | underico cabez.garcia#     | under p maldonado#         | under jain#                | underick j collins#      |
| undertaker         | underta       | undertall nix#             | undertan starlir#          | underta romero#            | undertala#                 | undertayshawn king#      |
| yokozuna           | yo            | yolanda y kryger#          | yosmanis a cruz#           | yovanny z bautista#        | yogesh chahar#             | yogesh chandra jo#       |
| yokozuna           | yoko          | yokondrae o hurdolk#       | yoko d chambers#           | yokonnihua f seplano#      | yoko#                      | yokoshania c carter#     |
| yokozuna           | yokozu        | yokozull bowell#           | yokozuma c adams#          | yokozua l uerda#           | yokozuddina#               | yokozua j harris#        |
| andre the giant    | and           | andre l arnold#            | andrew j denny#            | andres r guajardo#         | andhav#                    | andrew j brown#          |
| andre the giant    | andre t       | andre t mccray#            | andre t mccoy#             | andre t salazar#           | andre topta#               | andre t morgan#          |
| andre the giant    | andre the g   | andre the gilmer#          | andre the gorsom#          | andre the garcia#          | andre the gadar#           | andre the getti#         |
| big show           | bi            | billy r jr collier#        | billy j ball#              | billy e caballero reyes#   | birender kuma .r yadav#    | billy j banks#           |
| big show           | big           | big a mitchell#            | big p o carson#            | big mariano#               | big singh#                 | big nemi#                |
| big show           | big sh        | big shardley j jr atturwow | big shulte#                | big shergorio luz#         | big shan#                  | big sharma#              |
| hulk hogan         | hu            | hubert l hunt#             | hugh walthall#             | humberto m malagon#        | husainpreet kour#          | hugh a holt#             |
| hulk hogan         | hulk          | hulk  iv herrit#           | hulk leath#                | hulk rodriguez.gonzale#    | hulk mohd#                 | hulk  kumari#            |
| hulk hogan         | hulk ho       | hulk hornes#               | hulk howstie#              | hulk hoelles.maldonado#    | hulk holoo chand singh#    | hulk holu#               |
| picachu            | p             | patrick l lee#             | paul a branch#             | pedro j salazarlopez#      | parveen kaur . simi#       | paul j boyd#             |
| picachu            | pic           | picky edmores#             | pick d sansom#             | picer g melendez#          | picay . sonu#              | pick d moran#            |
| picachu            | picac         | picace lewis#              | picacco turdlere u alvie#  | picacio burgos#            | picachiram akwal#          | picace a james#          |
| mewtwo             | m             | marcus d cook#             | michael a mckinney#        | mario a martinez#          | manish kumar  mandal#      | mark a barnes#           |
| mewtwo             | mew           | mewille m rodriguez#       | mewald j donnynons#        | mewel achevedo#            | mewa aggarwal#             | mewasui . sanjali#       |
| mewtwo             | mewt          | mewtis f jr baken#         | mewthed shewnert#          | mewtel diaz#               | mewta#                     | mewtilles dison#         |
| jigglypuff         | ji            | jimmy l jr floyd#          | jimmy d booher#            | jimmy r torres#            | jitender sehrawat#         | jimmy l barnett#         |
| jigglypuff         | jiggl         | jiggl jordan#              | jigglan a layhorn#         | jigglan r alvaro#          | jiggla#                    | jiggl r porter#          |
| jigglypuff         | jigglyp       | jigglyphal#                | jigglyp t cedeno#          | jigglype vernarda#         | jigglypri . papa#          | jigglypu g martin#       |
| bulbasaur          | bu            | burnell mckenney#          | buddy l cook#              | bulfrano garcia#           | budh prakash#              | buddy l mccray#          |
| bulbasaur          | bulb          | bulber l hargus#           | bulbernon k manond#        | bulbim m rolon#            | bulbul#                    | bulbul#                  |
| bulbasaur          | bulbas        | bulbas  jr littles#        | bulbas.san s layvoro#      | bulbashumald gasialisanosh | bulbas#                    | bulbas b rivera#         |
| charizard          | ch            | charles e hall#            | christopher m brannon#     | christian rivera#          | chander shekhar#           | christopher m mcclendon# |
| charizard          | char          | charles e hall#            | charles e jr hartsfield#   | charles r murillo#         | charan singh . minchu#     | charles a brown#         |
| charizard          | chariz        | charizelo m laster#        | charizy a berry#           | chariz l martinez#         | chariz#                    | chariz c hill#           |
