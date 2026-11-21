//+------------------------------------------------------------------+
//| GoldBotSignals.mq5                                                |
//| Draws the gold_bot strategy's trades on the chart. Read only:     |
//| it never sends an order. Data comes from gold_bot_signals.csv in  |
//| the terminal's common Files folder, written by the Python app.    |
//| Draws: buy / sell arrows, exit marks, entry-to-exit lines (green  |
//| win, red loss, yellow still open), stop and target of an open     |
//| trade, and signal marks. Reloads by itself every 2 seconds.       |
//+------------------------------------------------------------------+
#property copyright "gold_bot"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

input string FileName    = "gold_bot_signals.csv";
input color  BuyColor    = clrDodgerBlue;
input color  SellColor   = clrTomato;
input color  WinColor    = clrLimeGreen;
input color  LossColor   = clrRed;
input color  OpenColor   = clrGold;
input bool   ShowLabels  = true;
input bool   ShowSignals = true;

const string PFX = "GBS_";
datetime g_stamp = 0;
long     g_size  = -1;
string   g_info  = "";

//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(2);
   Load();
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, PFX);
   Comment("");
  }
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated, const datetime &time[],
                const double &open[], const double &high[], const double &low[], const double &close[],
                const long &tick_volume[], const long &volume[], const int &spread[])
  {
   return(rates_total);
  }
//+------------------------------------------------------------------+
void OnTimer()
  {
   Load();
  }
//+------------------------------------------------------------------+
void Put(const string name, const ENUM_OBJECT type, const datetime t1, const double p1,
         const datetime t2 = 0, const double p2 = 0)
  {
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
   ObjectCreate(0, name, type, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }
//+------------------------------------------------------------------+
void Label(const string name, const datetime t, const double p, const string txt, const color c, const bool above)
  {
   if(!ShowLabels || txt == "")
      return;
   Put(name, OBJ_TEXT, t, p);
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, above ? ANCHOR_LEFT_LOWER : ANCHOR_LEFT_UPPER);
  }
//+------------------------------------------------------------------+
void Load()
  {
   int h = FileOpen(FileName, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(h == INVALID_HANDLE)
     {
      Comment("GoldBotSignals: waiting for ", FileName, " in the common Files folder");
      return;
     }
   datetime stamp = (datetime)FileGetInteger(h, FILE_MODIFY_DATE);
   long size = FileSize(h);
   if(stamp == g_stamp && size == g_size)
     {
      FileClose(h);
      return;
     }
   g_stamp = stamp;
   g_size = size;

   string head = FileReadString(h);           // #symbol,XAUUSD,updated,title
   string hp[];
   int hn = StringSplit(head, ',', hp);
   if(hn >= 2 && hp[1] != _Symbol && StringFind(_Symbol, hp[1]) < 0 && StringFind(hp[1], _Symbol) < 0)
     {
      FileClose(h);
      ObjectsDeleteAll(0, PFX);
      Comment("GoldBotSignals: the file is for ", hp[1], ", this chart is ", _Symbol);
      return;
     }
   string title = hn >= 4 ? hp[3] : "";
   FileReadString(h);                          // column names

   ObjectsDeleteAll(0, PFX);
   int n = 0, buys = 0, sells = 0, exits = 0;
   while(!FileIsEnding(h))
     {
      string line = FileReadString(h);
      if(StringLen(line) < 5)
         continue;
      string f[];
      if(StringSplit(line, ',', f) < 5)
         continue;
      // kind,time,price,side,text,time2,price2,win
      string kind = f[0];
      datetime t1 = StringToTime(f[1]);
      double p1 = StringToDouble(f[2]);
      bool isBuy = (f[3] == "BUY");
      string txt = f[4];
      datetime t2 = (ArraySize(f) > 5 && f[5] != "") ? StringToTime(f[5]) : 0;
      double p2 = (ArraySize(f) > 6 && f[6] != "") ? StringToDouble(f[6]) : 0;
      int win = (ArraySize(f) > 7 && f[7] != "") ? (int)StringToInteger(f[7]) : -1;
      string id = PFX + (string)n++;

      if(kind == "ENTRY")
        {
         Put(id, isBuy ? OBJ_ARROW_BUY : OBJ_ARROW_SELL, t1, p1);
         ObjectSetInteger(0, id, OBJPROP_COLOR, isBuy ? BuyColor : SellColor);
         ObjectSetInteger(0, id, OBJPROP_WIDTH, 2);
         ObjectSetString(0, id, OBJPROP_TOOLTIP, txt);
         Label(id + "t", t1, p1, txt, isBuy ? BuyColor : SellColor, !isBuy);
         if(isBuy) buys++; else sells++;
        }
      else if(kind == "EXIT")
        {
         Put(id, OBJ_ARROW, t1, p1);
         ObjectSetInteger(0, id, OBJPROP_ARROWCODE, 251);   // x mark
         color c = win == 1 ? WinColor : (win == 0 ? LossColor : OpenColor);
         ObjectSetInteger(0, id, OBJPROP_COLOR, c);
         ObjectSetInteger(0, id, OBJPROP_WIDTH, 2);
         ObjectSetString(0, id, OBJPROP_TOOLTIP, txt);
         Label(id + "t", t1, p1, txt, c, isBuy);
         exits++;
        }
      else if(kind == "LINE")
        {
         Put(id, OBJ_TREND, t1, p1, t2, p2);
         ObjectSetInteger(0, id, OBJPROP_COLOR, win == 1 ? WinColor : (win == 0 ? LossColor : OpenColor));
         ObjectSetInteger(0, id, OBJPROP_STYLE, STYLE_DOT);
         ObjectSetInteger(0, id, OBJPROP_RAY_RIGHT, false);
        }
      else if(kind == "SL" || kind == "TP")
        {
         Put(id, OBJ_TREND, t1, p1, t2, p2);
         color c = (kind == "SL") ? LossColor : WinColor;
         ObjectSetInteger(0, id, OBJPROP_COLOR, c);
         ObjectSetInteger(0, id, OBJPROP_STYLE, STYLE_DASH);
         ObjectSetInteger(0, id, OBJPROP_RAY_RIGHT, false);
         Label(id + "t", t2, p2, txt, c, true);
        }
      else if(kind == "SIG" && ShowSignals)
        {
         Put(id, OBJ_ARROW, t1, p1);
         ObjectSetInteger(0, id, OBJPROP_ARROWCODE, isBuy ? 233 : 234);
         ObjectSetInteger(0, id, OBJPROP_COLOR, isBuy ? BuyColor : SellColor);
         ObjectSetInteger(0, id, OBJPROP_ANCHOR, isBuy ? ANCHOR_TOP : ANCHOR_BOTTOM);
         ObjectSetString(0, id, OBJPROP_TOOLTIP, txt);
        }
     }
   FileClose(h);
   g_info = StringFormat("GoldBotSignals %s | %d buys, %d sells, %d exits | file updated %s", title, buys, sells, exits,
                         TimeToString(stamp, TIME_DATE | TIME_MINUTES | TIME_SECONDS));
   Comment(g_info);
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
