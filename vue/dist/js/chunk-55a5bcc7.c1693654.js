(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-55a5bcc7"],{"0ccb":function(t,a,i){var n=i("50c4"),e=i("577e"),o=i("1148"),c=i("1d80"),r=Math.ceil,l=function(t){return function(a,i,l){var u,s,d=e(c(a)),f=n(i),p=d.length,h=void 0===l?" ":e(l);return f<=p||""==h?d:(u=f-p,s=o.call(h,r(u/h.length)),s.length>u&&(s=s.slice(0,u)),t?d+s:s+d)}};t.exports={start:l(!1),end:l(!0)}},"4d90":function(t,a,i){"use strict";var n=i("23e7"),e=i("0ccb").start,o=i("9a0c");n({target:"String",proto:!0,forced:o},{padStart:function(t){return e(this,t,arguments.length>1?arguments[1]:void 0)}})},"7bf5":function(t,a,i){"use strict";i("bf44")},"9a0c":function(t,a,i){var n=i("342f");t.exports=/Version\/10(?:\.\d+){1,2}(?: [\w./]+)?(?: Mobile\/\w+)? Safari\//.test(n)},bf2e:function(t,a,i){"use strict";i.r(a);var n=function(){var t=this,a=t.$createElement,i=t._self._c||a;return i("div",{staticClass:"item"},[i("div",{staticClass:"audio"},[i("div",{staticClass:"audio__card"},[i("div",{class:["audio__btn",{pause:t.playing}],on:{click:t.handleClick}}),t.initial?i("div"):i("av-waveform",{key:t.update,attrs:{"canv-width":500,"canv-height":23,"played-line-color":"#65B9F4","noplayed-line-color":"#2B5278","played-line-width":0,playtime:!1,"canv-top":!0,"canv-class":"custom-player","canv-fill-color":"#2B5278","ref-link":"audio"}}),i("p",{staticClass:"audio__time"},[t._v(t._s(t.formatTime(t.curTime))+" / "+t._s(t.duration))]),i("audio",{ref:"audio",attrs:{src:t.src,preload:"none"},on:{canplay:t.canplay,timeupdate:function(a){t.curTime=a.target.currentTime},play:function(a){t.playing=!0},pause:function(a){t.playing=!1}}})],1)])])},e=[],o=(i("99af"),i("b680"),i("4d90"),{name:"AudioCard",props:{value:{type:String,default:""}},data:function(){return{loaded:!1,curTime:0,playing:!1,initial:!0}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(Date.now())},duration:function(){return this.loaded?this.formatTime(this.$refs.audio.duration):"00:00"}},methods:{handleClick:function(){var t=this;this.initial&&(this.initial=!1),setTimeout((function(){t.playing?t.$refs.audio.pause():t.$refs.audio.play()}),1)},canplay:function(){this.loaded=!0},formatTime:function(t){var a=Math.floor(t.toFixed()/60),i=t.toFixed()%60;return"".concat(String(a).padStart(2,"0"),":").concat(String(i).padStart(2,"0"))}},mounted:function(){this.$el.querySelector("audio").setAttribute("controlsList","nodownload noplaybackrate")}}),c=o,r=(i("7bf5"),i("2877")),l=Object(r["a"])(c,n,e,!1,null,null,null);a["default"]=l.exports},bf44:function(t,a,i){}}]);