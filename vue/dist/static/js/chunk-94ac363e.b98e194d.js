(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-94ac363e"],{"0ccb":function(t,a,i){var n=i("e330"),e=i("50c4"),r=i("577e"),c=i("1148"),o=i("1d80"),u=n(c),d=n("".slice),s=Math.ceil,l=function(t){return function(a,i,n){var c,l,f=r(o(a)),p=e(i),h=f.length,v=void 0===n?" ":r(n);return p<=h||""==v?f:(c=p-h,l=u(v,s(c/v.length)),l.length>c&&(l=d(l,0,c)),t?f+l:l+f)}};t.exports={start:l(!1),end:l(!0)}},"3d2d":function(t,a,i){"use strict";i("ae74")},"4d90":function(t,a,i){"use strict";var n=i("23e7"),e=i("0ccb").start,r=i("9a0c");n({target:"String",proto:!0,forced:r},{padStart:function(t){return e(this,t,arguments.length>1?arguments[1]:void 0)}})},9784:function(t,a,i){"use strict";i.r(a);var n=function(){var t=this,a=t.$createElement,i=t._self._c||a;return i("div",{staticClass:"item"},[i("div",{staticClass:"audio"},[i("div",{staticClass:"audio__card"},[i("div",{class:["audio__btn",{pause:t.playing}],on:{click:t.handleClick}}),t.initial?i("div"):i("av-bars",{attrs:{"canv-width":95,"canv-height":25,"bar-width":2,"bar-color":"#2B5278","canv-fill-color":"#242F3D","ref-link":"audio"}}),i("p",{staticClass:"audio__time"},[t._v(t._s(t.formatTime(t.curTime))+" / "+t._s(t.duration))]),i("audio",{ref:"audio",attrs:{src:t.src,preload:"none"},on:{canplay:t.canplay,timeupdate:function(a){t.curTime=a.target.currentTime},play:function(a){t.playing=!0},pause:function(a){t.playing=!1}}})],1)])])},e=[],r=(i("99af"),i("b680"),i("4d90"),{name:"TableAudio",props:{value:{type:String,default:""},update:{type:String,default:""}},data:function(){return{loaded:!1,curTime:0,playing:!1,initial:!0}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)},duration:function(){return this.loaded?this.formatTime(this.$refs.audio.duration):"00:00"}},methods:{handleClick:function(){var t=this;this.initial&&(this.initial=!1),setTimeout((function(){t.playing?t.$refs.audio.pause():t.$refs.audio.play()}),1)},canplay:function(){this.loaded=!0},formatTime:function(t){var a=Math.floor(t.toFixed()/60),i=t.toFixed()%60;return"".concat(String(a).padStart(2,"0"),":").concat(String(i).padStart(2,"0"))}}}),c=r,o=(i("3d2d"),i("2877")),u=Object(o["a"])(c,n,e,!1,null,"17a5a070",null);a["default"]=u.exports},"9a0c":function(t,a,i){var n=i("342f");t.exports=/Version\/10(?:\.\d+){1,2}(?: [\w./]+)?(?: Mobile\/\w+)? Safari\//.test(n)},ae74:function(t,a,i){}}]);