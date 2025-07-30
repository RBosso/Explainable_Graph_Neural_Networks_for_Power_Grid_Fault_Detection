# # +
#lines = """New Line.L1     Phases=1 Bus1=1.2        Bus2=2.2        LineCode=10   Length=0.175
#New Line.L2     Phases=1 Bus1=1.3        Bus2=3.3        LineCode=11   Length=0.25
#New Line.L3     Phases=3 Bus1=1.1.2.3    Bus2=7.1.2.3    LineCode=1    Length=0.3
#New Line.L4     Phases=1 Bus1=3.3        Bus2=4.3        LineCode=11   Length=0.2
#New Line.L5     Phases=1 Bus1=3.3        Bus2=5.3        LineCode=11   Length=0.325
#New Line.L6     Phases=1 Bus1=5.3        Bus2=6.3        LineCode=11   Length=0.25
#New Line.L7     Phases=3 Bus1=7.1.2.3    Bus2=8.1.2.3    LineCode=1    Length=0.2
#New Line.L8     Phases=1 Bus1=8.2        Bus2=12.2       LineCode=10   Length=0.225
#New Line.L9     Phases=1 Bus1=8.1        Bus2=9.1        LineCode=9    Length=0.225
#New Line.L10    Phases=3 Bus1=8.1.2.3    Bus2=13.1.2.3   LineCode=1    Length=0.3
#New Line.L11    Phases=1 Bus1=9r.1       Bus2=14.1       LineCode=9    Length=0.425
#New Line.L12    Phases=1 Bus1=13.3       Bus2=34.3       LineCode=11   Length=0.15
#New Line.L13    Phases=3 Bus1=13.1.2.3   Bus2=18.1.2.3   LineCode=2    Length=0.825
#New Line.L14    Phases=1 Bus1=14.1       Bus2=11.1       LineCode=9    Length=0.25
#New Line.L15    Phases=1 Bus1=14.1       Bus2=10.1       LineCode=9    Length=0.25
#New Line.L16    Phases=1 Bus1=15.3       Bus2=16.3       LineCode=11   Length=0.375
#New Line.L17    Phases=1 Bus1=15.3       Bus2=17.3       LineCode=11   Length=0.35
#New Line.L18    Phases=1 Bus1=18.1       Bus2=19.1       LineCode=9    Length=0.25
#New Line.L19    Phases=3 Bus1=18.1.2.3   Bus2=21.1.2.3   LineCode=2    Length=0.3
#New Line.L20    Phases=1 Bus1=19.1       Bus2=20.1       LineCode=9    Length=0.325
#New Line.L21    Phases=1 Bus1=21.2       Bus2=22.2       LineCode=10   Length=0.525
#New Line.L22    Phases=3 Bus1=21.1.2.3   Bus2=23.1.2.3   LineCode=2    Length=0.25
#New Line.L23    Phases=1 Bus1=23.3       Bus2=24.3       LineCode=11   Length=0.55
#New Line.L24    Phases=3 Bus1=23.1.2.3   Bus2=25.1.2.3   LineCode=2    Length=0.275
#New Line.L25    Phases=2 Bus1=25r.1.3    Bus2=26.1.3     LineCode=7    Length=0.35
#New Line.L26    Phases=3 Bus1=25.1.2.3   Bus2=28.1.2.3   LineCode=2    Length=0.2
#New Line.L27    Phases=2 Bus1=26.1.3     Bus2=27.1.3     LineCode=7    Length=0.275
#New Line.L28    Phases=1 Bus1=26.3       Bus2=31.3       LineCode=11   Length=0.225
#New Line.L29    Phases=1 Bus1=27.1       Bus2=33.1       LineCode=9    Length=0.5
#New Line.L30    Phases=3 Bus1=28.1.2.3   Bus2=29.1.2.3   LineCode=2    Length=0.3
#New Line.L31    Phases=3 Bus1=29.1.2.3   Bus2=30.1.2.3   LineCode=2    Length=0.35
#New Line.L32    Phases=3 Bus1=30.1.2.3   Bus2=250.1.2.3  LineCode=2    Length=0.2
#New Line.L33    Phases=1 Bus1=31.3       Bus2=32.3       LineCode=11   Length=0.3
#New Line.L34    Phases=1 Bus1=34.3       Bus2=15.3       LineCode=11   Length=0.1
#New Line.L35    Phases=2 Bus1=35.1.2     Bus2=36.1.2     LineCode=8    Length=0.65
#New Line.L36    Phases=3 Bus1=35.1.2.3   Bus2=40.1.2.3   LineCode=1    Length=0.25
#New Line.L37    Phases=1 Bus1=36.1       Bus2=37.1       LineCode=9    Length=0.3
#New Line.L38    Phases=1 Bus1=36.2       Bus2=38.2       LineCode=10   Length=0.25
#New Line.L39    Phases=1 Bus1=38.2       Bus2=39.2       LineCode=10   Length=0.325
#New Line.L40    Phases=1 Bus1=40.3       Bus2=41.3       LineCode=11   Length=0.325
#New Line.L41    Phases=3 Bus1=40.1.2.3   Bus2=42.1.2.3   LineCode=1    Length=0.25
#New Line.L42    Phases=1 Bus1=42.2       Bus2=43.2       LineCode=10   Length=0.5
#New Line.L43    Phases=3 Bus1=42.1.2.3   Bus2=44.1.2.3   LineCode=1    Length=0.2
#New Line.L44    Phases=1 Bus1=44.1       Bus2=45.1       LineCode=9    Length=0.2
#New Line.L45    Phases=3 Bus1=44.1.2.3   Bus2=47.1.2.3   LineCode=1    Length=0.25
#New Line.L46    Phases=1 Bus1=45.1       Bus2=46.1       LineCode=9    Length=0.3
#New Line.L47    Phases=3 Bus1=47.1.2.3   Bus2=48.1.2.3   LineCode=4    Length=0.15
#New Line.L48    Phases=3 Bus1=47.1.2.3   Bus2=49.1.2.3   LineCode=4    Length=0.25
#New Line.L49    Phases=3 Bus1=49.1.2.3   Bus2=50.1.2.3   LineCode=4    Length=0.25
#New Line.L50    Phases=3 Bus1=50.1.2.3   Bus2=51.1.2.3   LineCode=4    Length=0.25
#New Line.L51    Phases=3 Bus1=51.1.2.3   Bus2=151.1.2.3  LineCode=4    Length=0.5
#New Line.L52    Phases=3 Bus1=52.1.2.3   Bus2=53.1.2.3   LineCode=1    Length=0.2
#New Line.L53    Phases=3 Bus1=53.1.2.3   Bus2=54.1.2.3   LineCode=1    Length=0.125
#New Line.L54    Phases=3 Bus1=54.1.2.3   Bus2=55.1.2.3   LineCode=1    Length=0.275
#New Line.L55    Phases=3 Bus1=54.1.2.3   Bus2=57.1.2.3   LineCode=3    Length=0.35
#New Line.L56    Phases=3 Bus1=55.1.2.3   Bus2=56.1.2.3   LineCode=1    Length=0.275
#New Line.L57    Phases=1 Bus1=57.2       Bus2=58.2       LineCode=10   Length=0.25
#New Line.L58    Phases=3 Bus1=57.1.2.3   Bus2=60.1.2.3   LineCode=3    Length=0.75
#New Line.L59    Phases=1 Bus1=58.2       Bus2=59.2       LineCode=10   Length=0.25
#New Line.L60    Phases=3 Bus1=60.1.2.3   Bus2=61.1.2.3   LineCode=5    Length=0.55
#New Line.L61    Phases=3 Bus1=60.1.2.3   Bus2=62.1.2.3   LineCode=12   Length=0.25
#New Line.L62    Phases=3 Bus1=62.1.2.3   Bus2=63.1.2.3   LineCode=12   Length=0.175
#New Line.L63    Phases=3 Bus1=63.1.2.3   Bus2=64.1.2.3   LineCode=12   Length=0.35
#New Line.L64    Phases=3 Bus1=64.1.2.3   Bus2=65.1.2.3   LineCode=12   Length=0.425
#New Line.L65    Phases=3 Bus1=65.1.2.3   Bus2=66.1.2.3   LineCode=12   Length=0.325
#New Line.L66    Phases=1 Bus1=67.1       Bus2=68.1       LineCode=9    Length=0.2
#New Line.L67    Phases=3 Bus1=67.1.2.3   Bus2=72.1.2.3   LineCode=3    Length=0.275
#New Line.L68    Phases=3 Bus1=67.1.2.3   Bus2=97.1.2.3   LineCode=3    Length=0.25
#New Line.L69    Phases=1 Bus1=68.1       Bus2=69.1       LineCode=9    Length=0.275
#New Line.L70    Phases=1 Bus1=69.1       Bus2=70.1       LineCode=9    Length=0.325
#New Line.L71    Phases=1 Bus1=70.1       Bus2=71.1       LineCode=9    Length=0.275
#New Line.L72    Phases=1 Bus1=72.3       Bus2=73.3       LineCode=11   Length=0.275
#New Line.L73    Phases=3 Bus1=72.1.2.3   Bus2=76.1.2.3   LineCode=3    Length=0.2
#New Line.L74    Phases=1 Bus1=73.3       Bus2=74.3       LineCode=11   Length=0.35
#New Line.L75    Phases=1 Bus1=74.3       Bus2=75.3       LineCode=11   Length=0.4
#New Line.L76    Phases=3 Bus1=76.1.2.3   Bus2=77.1.2.3   LineCode=6    Length=0.4
#New Line.L77    Phases=3 Bus1=76.1.2.3   Bus2=86.1.2.3   LineCode=3    Length=0.7
#New Line.L78    Phases=3 Bus1=77.1.2.3   Bus2=78.1.2.3   LineCode=6    Length=0.1
#New Line.L79    Phases=3 Bus1=78.1.2.3   Bus2=79.1.2.3   LineCode=6    Length=0.225
#New Line.L80    Phases=3 Bus1=78.1.2.3   Bus2=80.1.2.3   LineCode=6    Length=0.475
#New Line.L81    Phases=3 Bus1=80.1.2.3   Bus2=81.1.2.3   LineCode=6    Length=0.175
#New Line.L82    Phases=3 Bus1=81.1.2.3   Bus2=82.1.2.3   LineCode=6    Length=0.25
#New Line.L83    Phases=1 Bus1=81.3       Bus2=84.3       LineCode=11   Length=0.675
#New Line.L84    Phases=3 Bus1=82.1.2.3   Bus2=83.1.2.3   LineCode=6    Length=0.25
#New Line.L85    Phases=1 Bus1=84.3       Bus2=85.3       LineCode=11   Length=0.475
#New Line.L86    Phases=3 Bus1=86.1.2.3   Bus2=87.1.2.3   LineCode=6    Length=0.45
#New Line.L87    Phases=1 Bus1=87.1       Bus2=88.1       LineCode=9    Length=0.175
#New Line.L88    Phases=3 Bus1=87.1.2.3   Bus2=89.1.2.3   LineCode=6    Length=0.275
#New Line.L89    Phases=1 Bus1=89.2       Bus2=90.2       LineCode=10   Length=0.25
#New Line.L90    Phases=3 Bus1=89.1.2.3   Bus2=91.1.2.3   LineCode=6    Length=0.225
#New Line.L91    Phases=1 Bus1=91.3       Bus2=92.3       LineCode=11   Length=0.3
#New Line.L92    Phases=3 Bus1=91.1.2.3   Bus2=93.1.2.3   LineCode=6    Length=0.225
#New Line.L93    Phases=1 Bus1=93.1       Bus2=94.1       LineCode=9    Length=0.275
#New Line.L94    Phases=3 Bus1=93.1.2.3   Bus2=95.1.2.3   LineCode=6    Length=0.3
#New Line.L95    Phases=1 Bus1=95.2       Bus2=96.2       LineCode=10   Length=0.2
#New Line.L96    Phases=3 Bus1=97.1.2.3   Bus2=98.1.2.3   LineCode=3    Length=0.275
#New Line.L97    Phases=3 Bus1=98.1.2.3   Bus2=99.1.2.3   LineCode=3    Length=0.55
#New Line.L98    Phases=3 Bus1=99.1.2.3   Bus2=100.1.2.3  LineCode=3    Length=0.3
#New Line.L99    Phases=3 Bus1=100.1.2.3  Bus2=450.1.2.3  LineCode=3    Length=0.8
#New Line.L100   Phases=1 Bus1=101.3      Bus2=102.3      LineCode=11   Length=0.225
#New Line.L101   Phases=3 Bus1=101.1.2.3  Bus2=105.1.2.3  LineCode=3    Length=0.275
#New Line.L102   Phases=1 Bus1=102.3      Bus2=103.3      LineCode=11   Length=0.325
#New Line.L103   Phases=1 Bus1=103.3      Bus2=104.3      LineCode=11   Length=0.7
#New Line.L104   Phases=1 Bus1=105.2      Bus2=106.2      LineCode=10   Length=0.225
#New Line.L105   Phases=3 Bus1=105.1.2.3  Bus2=108.1.2.3  LineCode=3    Length=0.325
#New Line.L106   Phases=1 Bus1=106.2      Bus2=107.2      LineCode=10   Length=0.575
#New Line.L107   Phases=1 Bus1=108.1      Bus2=109.1      LineCode=9    Length=0.45
#New Line.L108   Phases=3 Bus1=108.1.2.3  Bus2=300.1.2.3  LineCode=3    Length=1
#New Line.L109   Phases=1 Bus1=109.1      Bus2=110.1      LineCode=9    Length=0.3
#New Line.L110   Phases=1 Bus1=110.1      Bus2=111.1      LineCode=9    Length=0.575
#New Line.L111   Phases=1 Bus1=110.1      Bus2=112.1      LineCode=9    Length=0.125
#New Line.L112   Phases=1 Bus1=112.1      Bus2=113.1      LineCode=9    Length=0.525
#New Line.L113   Phases=1 Bus1=113.1      Bus2=114.1      LineCode=9    Length=0.325
#New Line.L114   Phases=3 Bus1=135.1.2.3  Bus2=35.1.2.3   LineCode=4    Length=0.375
#New Line.L115   Phases=3 Bus1=149.1.2.3  Bus2=1.1.2.3    LineCode=1    Length=0.4
#New Line.L116   Phases=3 Bus1=152.1.2.3  Bus2=52.1.2.3   LineCode=1    Length=0.4
#New Line.L117   Phases=3 Bus1=160r.1.2.3 Bus2=67.1.2.3   LineCode=6    Length=0.35
#New Line.L118   Phases=3 Bus1=197.1.2.3  Bus2=101.1.2.3  LineCode=3    Length=0.25
#New Line.Sw1    phases=3  Bus1=150r   Bus2=149    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
#New Line.Sw2    phases=3  Bus1=13     Bus2=152    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
#New Line.Sw3    phases=3  Bus1=18     Bus2=135    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
#New Line.Sw4    phases=3  Bus1=60     Bus2=160    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
#New Line.Sw5    phases=3  Bus1=97     Bus2=197    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
#New Line.Sw6    phases=3  Bus1=61     Bus2=61s    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
#New Line.Sw6    phases=3  Bus1=61s     Bus2=610    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001"""
import numpy as np
#
#lines = lines.splitlines()
## print(lines)
#print(len(lines))
#
#lines_list = []
#for l in lines:
#    # print(l)
#    e = l.split()[1:5]
#    e.append(l.split()[-1])
#    print(e)
#    lines_list.append(e)

lines = """New Line.L1     Phases=1 Bus1=1.2        Bus2=2.2        LineCode=10   Length=0.175
New Line.L2     Phases=1 Bus1=1.3        Bus2=3.3        LineCode=11   Length=0.25
New Line.L3     Phases=3 Bus1=1.1.2.3    Bus2=7.1.2.3    LineCode=1    Length=0.3
New Line.L4     Phases=1 Bus1=3.3        Bus2=4.3        LineCode=11   Length=0.2
New Line.L5     Phases=1 Bus1=3.3        Bus2=5.3        LineCode=11   Length=0.325
New Line.L6     Phases=1 Bus1=5.3        Bus2=6.3        LineCode=11   Length=0.25
New Line.L7     Phases=3 Bus1=7.1.2.3    Bus2=8.1.2.3    LineCode=1    Length=0.2
New Line.L8     Phases=1 Bus1=8.2        Bus2=12.2       LineCode=10   Length=0.225
New Line.L9     Phases=1 Bus1=8.1        Bus2=9.1        LineCode=9    Length=0.225
New Line.L10    Phases=3 Bus1=8.1.2.3    Bus2=13.1.2.3   LineCode=1    Length=0.3
New Line.L11    Phases=1 Bus1=9r.1       Bus2=14.1       LineCode=9    Length=0.425
New Line.L12    Phases=1 Bus1=13.3       Bus2=34.3       LineCode=11   Length=0.15
New Line.L13    Phases=3 Bus1=13.1.2.3   Bus2=18.1.2.3   LineCode=2    Length=0.825
New Line.L14    Phases=1 Bus1=14.1       Bus2=11.1       LineCode=9    Length=0.25
New Line.L15    Phases=1 Bus1=14.1       Bus2=10.1       LineCode=9    Length=0.25
New Line.L16    Phases=1 Bus1=15.3       Bus2=16.3       LineCode=11   Length=0.375
New Line.L17    Phases=1 Bus1=15.3       Bus2=17.3       LineCode=11   Length=0.35
New Line.L18    Phases=1 Bus1=18.1       Bus2=19.1       LineCode=9    Length=0.25
New Line.L19    Phases=3 Bus1=18.1.2.3   Bus2=21.1.2.3   LineCode=2    Length=0.3
New Line.L20    Phases=1 Bus1=19.1       Bus2=20.1       LineCode=9    Length=0.325
New Line.L21    Phases=1 Bus1=21.2       Bus2=22.2       LineCode=10   Length=0.525
New Line.L22    Phases=3 Bus1=21.1.2.3   Bus2=23.1.2.3   LineCode=2    Length=0.25
New Line.L23    Phases=1 Bus1=23.3       Bus2=24.3       LineCode=11   Length=0.55
New Line.L24    Phases=3 Bus1=23.1.2.3   Bus2=25.1.2.3   LineCode=2    Length=0.275
New Line.L25    Phases=2 Bus1=25r.1.3    Bus2=26.1.3     LineCode=7    Length=0.35
New Line.L26    Phases=3 Bus1=25.1.2.3   Bus2=28.1.2.3   LineCode=2    Length=0.2
New Line.L27    Phases=2 Bus1=26.1.3     Bus2=27.1.3     LineCode=7    Length=0.275
New Line.L28    Phases=1 Bus1=26.3       Bus2=31.3       LineCode=11   Length=0.225
New Line.L29    Phases=1 Bus1=27.1       Bus2=33.1       LineCode=9    Length=0.5
New Line.L30    Phases=3 Bus1=28.1.2.3   Bus2=29.1.2.3   LineCode=2    Length=0.3
New Line.L31    Phases=3 Bus1=29.1.2.3   Bus2=30.1.2.3   LineCode=2    Length=0.35
New Line.L32    Phases=3 Bus1=30.1.2.3   Bus2=250.1.2.3  LineCode=2    Length=0.2
New Line.L33    Phases=1 Bus1=31.3       Bus2=32.3       LineCode=11   Length=0.3
New Line.L34    Phases=1 Bus1=34.3       Bus2=15.3       LineCode=11   Length=0.1
New Line.L35    Phases=2 Bus1=35.1.2     Bus2=36.1.2     LineCode=8    Length=0.65
New Line.L36    Phases=3 Bus1=35.1.2.3   Bus2=40.1.2.3   LineCode=1    Length=0.25
New Line.L37    Phases=1 Bus1=36.1       Bus2=37.1       LineCode=9    Length=0.3
New Line.L38    Phases=1 Bus1=36.2       Bus2=38.2       LineCode=10   Length=0.25
New Line.L39    Phases=1 Bus1=38.2       Bus2=39.2       LineCode=10   Length=0.325
New Line.L40    Phases=1 Bus1=40.3       Bus2=41.3       LineCode=11   Length=0.325
New Line.L41    Phases=3 Bus1=40.1.2.3   Bus2=42.1.2.3   LineCode=1    Length=0.25
New Line.L42    Phases=1 Bus1=42.2       Bus2=43.2       LineCode=10   Length=0.5
New Line.L43    Phases=3 Bus1=42.1.2.3   Bus2=44.1.2.3   LineCode=1    Length=0.2
New Line.L44    Phases=1 Bus1=44.1       Bus2=45.1       LineCode=9    Length=0.2
New Line.L45    Phases=3 Bus1=44.1.2.3   Bus2=47.1.2.3   LineCode=1    Length=0.25
New Line.L46    Phases=1 Bus1=45.1       Bus2=46.1       LineCode=9    Length=0.3
New Line.L47    Phases=3 Bus1=47.1.2.3   Bus2=48.1.2.3   LineCode=4    Length=0.15
New Line.L48    Phases=3 Bus1=47.1.2.3   Bus2=49.1.2.3   LineCode=4    Length=0.25
New Line.L49    Phases=3 Bus1=49.1.2.3   Bus2=50.1.2.3   LineCode=4    Length=0.25
New Line.L50    Phases=3 Bus1=50.1.2.3   Bus2=51.1.2.3   LineCode=4    Length=0.25
New Line.L51    Phases=3 Bus1=51.1.2.3   Bus2=151.1.2.3  LineCode=4    Length=0.5
New Line.L52    Phases=3 Bus1=52.1.2.3   Bus2=53.1.2.3   LineCode=1    Length=0.2
New Line.L53    Phases=3 Bus1=53.1.2.3   Bus2=54.1.2.3   LineCode=1    Length=0.125
New Line.L54    Phases=3 Bus1=54.1.2.3   Bus2=55.1.2.3   LineCode=1    Length=0.275
New Line.L55    Phases=3 Bus1=54.1.2.3   Bus2=57.1.2.3   LineCode=3    Length=0.35
New Line.L56    Phases=3 Bus1=55.1.2.3   Bus2=56.1.2.3   LineCode=1    Length=0.275
New Line.L57    Phases=1 Bus1=57.2       Bus2=58.2       LineCode=10   Length=0.25
New Line.L58    Phases=3 Bus1=57.1.2.3   Bus2=60.1.2.3   LineCode=3    Length=0.75
New Line.L59    Phases=1 Bus1=58.2       Bus2=59.2       LineCode=10   Length=0.25
New Line.L60    Phases=3 Bus1=60.1.2.3   Bus2=61.1.2.3   LineCode=5    Length=0.55
New Line.L61    Phases=3 Bus1=60.1.2.3   Bus2=62.1.2.3   LineCode=12   Length=0.25
New Line.L62    Phases=3 Bus1=62.1.2.3   Bus2=63.1.2.3   LineCode=12   Length=0.175
New Line.L63    Phases=3 Bus1=63.1.2.3   Bus2=64.1.2.3   LineCode=12   Length=0.35
New Line.L64    Phases=3 Bus1=64.1.2.3   Bus2=65.1.2.3   LineCode=12   Length=0.425
New Line.L65    Phases=3 Bus1=65.1.2.3   Bus2=66.1.2.3   LineCode=12   Length=0.325
New Line.L66    Phases=1 Bus1=67.1       Bus2=68.1       LineCode=9    Length=0.2
New Line.L67    Phases=3 Bus1=67.1.2.3   Bus2=72.1.2.3   LineCode=3    Length=0.275
New Line.L68    Phases=3 Bus1=67.1.2.3   Bus2=97.1.2.3   LineCode=3    Length=0.25
New Line.L69    Phases=1 Bus1=68.1       Bus2=69.1       LineCode=9    Length=0.275
New Line.L70    Phases=1 Bus1=69.1       Bus2=70.1       LineCode=9    Length=0.325
New Line.L71    Phases=1 Bus1=70.1       Bus2=71.1       LineCode=9    Length=0.275
New Line.L72    Phases=1 Bus1=72.3       Bus2=73.3       LineCode=11   Length=0.275
New Line.L73    Phases=3 Bus1=72.1.2.3   Bus2=76.1.2.3   LineCode=3    Length=0.2
New Line.L74    Phases=1 Bus1=73.3       Bus2=74.3       LineCode=11   Length=0.35
New Line.L75    Phases=1 Bus1=74.3       Bus2=75.3       LineCode=11   Length=0.4
New Line.L76    Phases=3 Bus1=76.1.2.3   Bus2=77.1.2.3   LineCode=6    Length=0.4
New Line.L77    Phases=3 Bus1=76.1.2.3   Bus2=86.1.2.3   LineCode=3    Length=0.7
New Line.L78    Phases=3 Bus1=77.1.2.3   Bus2=78.1.2.3   LineCode=6    Length=0.1
New Line.L79    Phases=3 Bus1=78.1.2.3   Bus2=79.1.2.3   LineCode=6    Length=0.225
New Line.L80    Phases=3 Bus1=78.1.2.3   Bus2=80.1.2.3   LineCode=6    Length=0.475
New Line.L81    Phases=3 Bus1=80.1.2.3   Bus2=81.1.2.3   LineCode=6    Length=0.175
New Line.L82    Phases=3 Bus1=81.1.2.3   Bus2=82.1.2.3   LineCode=6    Length=0.25
New Line.L83    Phases=1 Bus1=81.3       Bus2=84.3       LineCode=11   Length=0.675
New Line.L84    Phases=3 Bus1=82.1.2.3   Bus2=83.1.2.3   LineCode=6    Length=0.25
New Line.L85    Phases=1 Bus1=84.3       Bus2=85.3       LineCode=11   Length=0.475
New Line.L86    Phases=3 Bus1=86.1.2.3   Bus2=87.1.2.3   LineCode=6    Length=0.45
New Line.L87    Phases=1 Bus1=87.1       Bus2=88.1       LineCode=9    Length=0.175
New Line.L88    Phases=3 Bus1=87.1.2.3   Bus2=89.1.2.3   LineCode=6    Length=0.275
New Line.L89    Phases=1 Bus1=89.2       Bus2=90.2       LineCode=10   Length=0.25
New Line.L90    Phases=3 Bus1=89.1.2.3   Bus2=91.1.2.3   LineCode=6    Length=0.225
New Line.L91    Phases=1 Bus1=91.3       Bus2=92.3       LineCode=11   Length=0.3
New Line.L92    Phases=3 Bus1=91.1.2.3   Bus2=93.1.2.3   LineCode=6    Length=0.225
New Line.L93    Phases=1 Bus1=93.1       Bus2=94.1       LineCode=9    Length=0.275
New Line.L94    Phases=3 Bus1=93.1.2.3   Bus2=95.1.2.3   LineCode=6    Length=0.3
New Line.L95    Phases=1 Bus1=95.2       Bus2=96.2       LineCode=10   Length=0.2
New Line.L96    Phases=3 Bus1=97.1.2.3   Bus2=98.1.2.3   LineCode=3    Length=0.275
New Line.L97    Phases=3 Bus1=98.1.2.3   Bus2=99.1.2.3   LineCode=3    Length=0.55
New Line.L98    Phases=3 Bus1=99.1.2.3   Bus2=100.1.2.3  LineCode=3    Length=0.3
New Line.L99    Phases=3 Bus1=100.1.2.3  Bus2=450.1.2.3  LineCode=3    Length=0.8
New Line.L100   Phases=1 Bus1=101.3      Bus2=102.3      LineCode=11   Length=0.225
New Line.L101   Phases=3 Bus1=101.1.2.3  Bus2=105.1.2.3  LineCode=3    Length=0.275
New Line.L102   Phases=1 Bus1=102.3      Bus2=103.3      LineCode=11   Length=0.325
New Line.L103   Phases=1 Bus1=103.3      Bus2=104.3      LineCode=11   Length=0.7
New Line.L104   Phases=1 Bus1=105.2      Bus2=106.2      LineCode=10   Length=0.225
New Line.L105   Phases=3 Bus1=105.1.2.3  Bus2=108.1.2.3  LineCode=3    Length=0.325
New Line.L106   Phases=1 Bus1=106.2      Bus2=107.2      LineCode=10   Length=0.575
New Line.L107   Phases=1 Bus1=108.1      Bus2=109.1      LineCode=9    Length=0.45
New Line.L108   Phases=3 Bus1=108.1.2.3  Bus2=300.1.2.3  LineCode=3    Length=1
New Line.L109   Phases=1 Bus1=109.1      Bus2=110.1      LineCode=9    Length=0.3
New Line.L110   Phases=1 Bus1=110.1      Bus2=111.1      LineCode=9    Length=0.575
New Line.L111   Phases=1 Bus1=110.1      Bus2=112.1      LineCode=9    Length=0.125
New Line.L112   Phases=1 Bus1=112.1      Bus2=113.1      LineCode=9    Length=0.525
New Line.L113   Phases=1 Bus1=113.1      Bus2=114.1      LineCode=9    Length=0.325
New Line.L114   Phases=3 Bus1=135.1.2.3  Bus2=35.1.2.3   LineCode=4    Length=0.375
New Line.L115   Phases=3 Bus1=149.1.2.3  Bus2=1.1.2.3    LineCode=1    Length=0.4
New Line.L116   Phases=3 Bus1=152.1.2.3  Bus2=52.1.2.3   LineCode=1    Length=0.4
New Line.L117   Phases=3 Bus1=160r.1.2.3 Bus2=67.1.2.3   LineCode=6    Length=0.35
New Line.L118   Phases=3 Bus1=197.1.2.3  Bus2=101.1.2.3  LineCode=3    Length=0.25
New Line.Sw1    phases=3  Bus1=150r   Bus2=149    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw1    phases=3  Bus1=150   Bus2=150r    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw2    phases=3  Bus1=13     Bus2=152    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw3    phases=3  Bus1=18     Bus2=135    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw4    phases=3  Bus1=60     Bus2=160    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw5    phases=3  Bus1=97     Bus2=197    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw6    phases=3  Bus1=61     Bus2=61s    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001
New Line.Sw6    phases=3  Bus1=61s     Bus2=610    r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001"""

# Added Buses 150 and 
import numpy as np
import sys

lines = lines.splitlines()
# print(lines)
print(len(lines))

lines_list = []
for l in lines:
    # print(l)
    e = l.split()[3:5]
    e = [e[0].split('=')[1].split('.')[0],e[1].split('=')[1].split('.')[0]]
#    e.append(l.split()[-1])
    # print(e)
    lines_list.append(e)

print(lines_list)

Buses = ['150','150R','149','1','2','3','7','4','5','6','8','12','9','13','9R',
 '14','34','18','11','10','15','16','17','19','21','20','22','23','24',
 '25','25R','26','28','27','31','33','29','30','250','32','35','36','40',
 '37','38','39','41','42','43','44','45','47','46','48','49','50','51',
 '151','52','53','54','55','57','56','58','60','59','61','62','63','64',
 '65','66','67','68','72','97','69','70','71','73','76','74','75','77',
 '86','78','79','80','81','82','84','83','85','87','88','89','90','91',
 '92','93','94','95','96','98','99','100','450','197','101','102','105',
 '103','104','106','108','107','109','300','110','111','112','113','114',
 '135','152','160R','160','61S','610']

X_axis = Buses
Y_axis = Buses
print(Buses)

AdjacencyMatrix = np.zeros((len(X_axis),len(X_axis)))
#print(X_axis)
# #print(AdjacencyMatrix)
X_count = 0
Y_count = 0
#
for X in X_axis:
    Y_count = 0
    for Y in Y_axis:
        # #print(X,Y)
        # #print(X_count, Y_count)
        # Y_count +=1
        # #print(X,Y)
        if ([X.lower(),Y.lower()] in lines_list) and X != Y:
            #print(X,Y)
            AdjacencyMatrix[Y_count][X_count] = 1
            AdjacencyMatrix[X_count][Y_count] = 1
        Y_count +=1
    X_count+=1

# #print(['Bus1', 'Bus2'] in graphdata)

#np.set_printoptions(threshold=sys.maxsize)
print(AdjacencyMatrix)
print(AdjacencyMatrix.shape)
 # #print(AdjacencyMatrix == AdjacencyMatrix.T)
if (False not in (AdjacencyMatrix == AdjacencyMatrix.T)):
    print("Adjacency Matrix is Symmetrical")
else:
    print("Not Symmetrical")

encoder = Buses

print(encoder)
print()

for pair in range(len(lines_list)):
    #    print(lines_list[pair][0])
#    print(lines_list[pair][1])    #print(pair))
    lines_list[pair][0] = encoder.index(lines_list[pair][0].upper())
    lines_list[pair][1] = encoder.index(lines_list[pair][1].upper())
removal = []
for pair in range(len(lines_list)):
    if lines_list[pair][0] == lines_list[pair][1]:
        removal.append(pair)
for r in reversed(removal):
    lines_list.pop(r)
reversed = []
for pair in lines_list:
    reversed.append([pair[1],pair[0]])
lines_list = lines_list + reversed

#Removes Duplicates from lines_list list
no_duplicates = []
for data in lines_list:
    if data not in no_duplicates:
        no_duplicates.append(data)
print(no_duplicates)
print(len(no_duplicates))
#print("T"*400)
