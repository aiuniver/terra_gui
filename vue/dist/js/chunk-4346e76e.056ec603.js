(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-4346e76e"],{1039:function(t,e,r){"use strict";r.r(e);var n=function(){var t=this,e=t.$createElement,r=t._self._c||e;return r("div",{staticClass:"t-text-segmented",style:{width:t.block_width}},[r("div",{staticClass:"t-text-segmented__content"},t._l(t.arrText,(function(e,n){var a=e.tags,l=e.word;return r("div",{key:"word_"+n,staticClass:"t-text-segmented__word"},[a.includes("p1")?r("div",{staticClass:"t-text-segmented__text"},[t._v(t._s(l))]):r("at-tooltip",[r("div",{staticClass:"t-text-segmented__text"},[t._v(t._s(l))]),r("template",{slot:"content"},t._l(a,(function(e){return r("div",{key:"colors_"+e,staticClass:"t-text-segmented__colors",style:"background-color: rgb("+t.rgb(e)+");"},[t._v(t._s(e))])})),0)],2),t._l(a,(function(e,a){return r("div",{key:"tags_"+n+"_"+a,staticClass:"t-text-segmented__line",style:"background-color: rgb("+t.rgb(e)+");"})}))],2)})),0)])},a=[],l=(r("5b81"),r("ac1f"),r("466d"),r("d81d"),r("a15b"),r("5319"),r("1276"),{name:"TableTextSegmented",props:{value:{type:String,default:""},color_mark:{type:Array,default:function(){return[]}},tags_color:{type:Object,default:function(){}},layer:{type:String,default:""},block_width:{type:String,default:"400px"}},computed:{tags:function(){var t;return(null===(t=this.tags_color)||void 0===t?void 0:t[this.layer])||{}},arrText:function(){var t=this,e=this.value.replaceAll(" ",""),r=e.match(/(<[sp][0-9]>)+([^<\/>]+)(<\/[sp][0-9]>)+/g);return r.map((function(e){return t.convert(e)}))}},methods:{rgb:function(t){var e=this.tags[t]||[];return e.join(" ")},convert:function(t){t=t.replace(/(<\/[^>]+>)+/g,"");var e=t.replace(/(<[^>]+>)+/g,"");t=t.replace(/></g,",");var r=t.match(/<(.+)>/)[1].split(",");return{tags:r,word:e}}}}),s=l,c=(r("ca51"),r("2877")),o=Object(c["a"])(s,n,a,!1,null,"be918f14",null);e["default"]=o.exports},"5b81":function(t,e,r){"use strict";var n=r("23e7"),a=r("1d80"),l=r("1626"),s=r("44e7"),c=r("577e"),o=r("dc4a"),i=r("ad6d"),u=r("0cb2"),d=r("b622"),g=r("c430"),p=d("replace"),f=RegExp.prototype,_=Math.max,v=function(t,e,r){return r>t.length?-1:""===e?r:t.indexOf(e,r)};n({target:"String",proto:!0},{replaceAll:function(t,e){var r,n,d,b,h,x,m,w,y,k=a(this),C=0,S=0,T="";if(null!=t){if(r=s(t),r&&(n=c(a("flags"in f?t.flags:i.call(t))),!~n.indexOf("g")))throw TypeError("`.replaceAll` does not allow non-global regexes");if(d=o(t,p),d)return d.call(t,k,e);if(g&&r)return c(k).replace(t,e)}b=c(k),h=c(t),x=l(e),x||(e=c(e)),m=h.length,w=_(1,m),C=v(b,h,0);while(-1!==C)y=x?c(e(h,C,b)):u(h,b,C,[],void 0,e),T+=b.slice(S,C)+y,S=C+m,C=v(b,h,C+w);return S<b.length&&(T+=b.slice(S)),T}})},b022:function(t,e,r){},ca51:function(t,e,r){"use strict";r("b022")}}]);