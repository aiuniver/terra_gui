(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-5a7f48d3"],{"07ce":function(t,a,e){},"0da8":function(t,a,e){},"2ba2":function(t,a,e){"use strict";e("b249")},"3af9":function(t,a,e){"use strict";e.r(a);var i=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",{staticClass:"card"},[e("div",{staticClass:"card__content"},["image_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("ImgCard",{attrs:{imgUrl:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"224px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"text_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.source))])],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1),"audio_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("AudioCard",{attrs:{value:t.card.source,update:t.RandId}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e()]):t._e(),"text_segmentation"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.format))])],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.card.format))])],1)]):t._e(),"audio_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("AudioCard",{attrs:{value:t.card.source,update:t.RandId}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"image_segmentation"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("ImgCard",{attrs:{imgUrl:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("ImgCard",{attrs:{imgUrl:t.card.segment}})],1)]):t._e(),"graphic"==t.type?e("div",{staticClass:"card__graphic"},[e("Plotly",{attrs:{data:t.card.data,layout:t.layout,"display-mode-bar":!1}})],1):t._e()]),e("div",{staticClass:"card__reload"},[e("button",{staticClass:"btn-reload",on:{click:t.ReloadCard}},[e("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])])])},s=[],c=e("5530"),r=(e("a9e3"),e("d3b7"),e("25f0"),e("4e82"),e("99af"),function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",{staticClass:"img-card"},[e("img",{staticClass:"img-card__image",attrs:{src:t.src}})])}),d=[],l=e("2f62"),n={name:"ImgCard",props:{imgUrl:{type:String,default:"img.png"}},computed:Object(c["a"])(Object(c["a"])({},Object(l["b"])({id:"deploy/getRandId"})),{},{src:function(){return"/_media/blank/?path=".concat(this.imgUrl,"&r=").concat(this.id)}})},o=n,u=(e("7e8a"),e("2877")),p=Object(u["a"])(o,r,d,!1,null,"5ab7da04",null),_=p.exports,f=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",{staticClass:"text-card"},[e("scrollbar",{attrs:{ops:t.ops}},[e("pre",[t._t("default",(function(){return[t._v("TEXT")]}))],2)])],1)},h=[],g={name:"TextCard",props:{text:{type:String,default:"text"}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},v=g,y=(e("f4ce"),Object(u["a"])(v,f,h,!1,null,"6e13d9d6",null)),C=y.exports,m=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",{staticClass:"item"},[e("div",{staticClass:"audio"},[e("div",{staticClass:"audio__card"},[e("av-waveform",{attrs:{"canv-width":500,"canv-height":23,"played-line-color":"#65B9F4","noplayed-line-color":"#2B5278","played-line-width":0,playtime:!1,"canv-top":!0,"canv-class":"custom-player","canv-fill-color":"#2B5278","audio-src":t.src}})],1)])])},x=[],b={name:"AudioCard",props:{value:{type:String,default:""},update:{type:String,default:""}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)}},mounted:function(){this.$el.querySelector("audio").setAttribute("controlsList","nodownload")}},w=b,T=(e("2ba2"),Object(u["a"])(w,m,x,!1,null,"04c83e76",null)),O=T.exports,R=e("04d11"),j={name:"IndexCard",components:{ImgCard:_,TextCard:C,Plotly:R["Plotly"],AudioCard:O},data:function(){return{}},props:{card:{type:Object,default:function(){return{}}},index:[String,Number]},methods:{ReloadCard:function(){this.$emit("reload",[this.index.toString()])}},mounted:function(){console.log(this.card),console.log(this.type)},computed:Object(c["a"])(Object(c["a"])({},Object(l["b"])({graphicData:"deploy/getGraphicData",defaultLayout:"deploy/getDefaultLayout",origTextStyle:"deploy/getOrigTextStyle",type:"deploy/getDeployType",RandId:"deploy/getRandId"})),{},{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},ClassificationResult:function(){var t=this.card.data,a="";t.sort((function(t,a){return t[1]<a[1]?1:-1}));for(var e=0;e<t.length;e++)a+="".concat(t[e][0]," - ").concat(t[e][1],"% \n");return a}})},I=j,S=(e("72ca"),Object(u["a"])(I,i,s,!1,null,"25f141b3",null));a["default"]=S.exports},"49bc":function(t,a,e){},"72ca":function(t,a,e){"use strict";e("49bc")},"7e8a":function(t,a,e){"use strict";e("0da8")},b249:function(t,a,e){},f4ce:function(t,a,e){"use strict";e("07ce")}}]);