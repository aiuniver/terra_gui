(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-184b284c"],{"303f":function(t,e,s){},e344:function(t,e,s){"use strict";s("303f")},f6b1:function(t,e,s){"use strict";s.r(e);var a=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"board"},[s("div",{staticClass:"wrapper"},[s("Filters"),s("div",{staticClass:"project-datasets-block datasets",style:t.height},[s("div",{staticClass:"title",on:{click:function(e){return t.click("name")}}},[t._v("Выберите датасет")]),s("scrollbar",[s("div",{staticClass:"inner"},[s("div",{staticClass:"dataset-card-container"},[s("div",{staticClass:"dataset-card-wrapper"},[t._l(t.datasets,(function(e,a){return[s("CardDataset",{key:a,attrs:{dataset:e,cardIndex:a,loaded:t.isLoaded(e)},on:{click:t.click,remove:t.remove}})]}))],2)])])])],1)],1)])},i=[],n=s("1da1"),c=s("5530"),r=(s("96cf"),s("b0c0"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{ref:"filters",staticClass:"project-datasets-block filters"},[s("div",{staticClass:"title"},[t._v("Теги")]),s("div",{staticClass:"inner"},[s("ul",t._l(t.tags,(function(e,a){var i=e.name,n=e.alias,c=e.active;return s("li",{key:a,class:{active:c},on:{click:function(e){return t.click(a,n)}}},[s("span",[t._v(" "+t._s(i)+" ")])])})),0)])])}),d=[],o={computed:{tags:{set:function(t){this.$store.dispatch("datasets/setTags",t)},get:function(){return this.$store.getters["datasets/getTags"]}},tagsFilter:{set:function(t){this.$store.dispatch("datasets/setTagsFilter",t)},get:function(){return this.$store.getters["datasets/getTagsFilter"]}}},methods:{click:function(t){this.tags[t].active=!this.tags[t].active,this.tagsFilter=this.tags.reduce((function(t,e){var s=e.active,a=e.alias;return s&&t.push(a),t}),[])},myEventHandler:function(){this.$store.dispatch("settings/setHeight",{filter:this.$refs.filters.clientHeight})}},watch:{tags:function(){var t=this;this.$nextTick((function(){t.$store.dispatch("settings/setHeight",{filter:t.$refs.filters.clientHeight})}))}},mounted:function(){var t=this;setTimeout((function(){t.$store.dispatch("settings/setHeight",{filter:t.$refs.filters.clientHeight})}),100)},created:function(){window.addEventListener("resize",this.myEventHandler)},destroyed:function(){window.removeEventListener("resize",this.myEventHandler)}},l=o,u=s("2877"),h=Object(u["a"])(l,r,d,!1,null,null,null),v=h.exports,f=s("2f62"),p=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{class:["dataset-card-item",{active:t.dataset.active&&!t.loaded,selected:t.loaded}],attrs:{title:t.dataset.name}},[s("div",{staticClass:"dataset-card",on:{click:function(e){return e.stopPropagation(),t.$emit("click",t.dataset,t.cardIndex)}}},[s("div",{staticClass:"card-title"},[t._v(t._s(t.dataset.name))]),s("div",{staticClass:"card-body",on:{click:t.click}},[t.dataset.tags.length<=4?s("div",t._l(t.dataset.tags,(function(e,a){var i=e.name;return s("div",{key:"tag_"+a,staticClass:"card-tag"},[t._v(" "+t._s(i)+" ")])})),0):s("scrollbar",{attrs:{ops:{bar:{background:"#17212b"}}}},t._l(t.dataset.tags,(function(e,a){var i=e.name;return s("div",{key:"tag_"+a,staticClass:"card-tag"},[t._v(" "+t._s(i)+" ")])})),0)],1),s("div",{class:"card-extra "+(t.dataset.size?"is-custom":"")},[s("div",{staticClass:"wrapper"},[s("span",[t._v(" "+t._s(t.dataset.size?t.dataset.size.short.toFixed(2)+" "+t.dataset.size.unit:"Предустановленный")+" ")])]),s("div",{staticClass:"remove",on:{click:function(e){return e.stopPropagation(),t.$emit("remove",t.dataset)}}})])])])},g=[],m=(s("a9e3"),{props:{dataset:{type:Object,default:function(){}},cardIndex:{type:Number},loaded:{type:Boolean,default:!1}},data:function(){return{index:0}},computed:{},methods:{click:function(){this.dataset.tags.length>this.index+4?this.index=this.index+4:this.index=0}}}),k=m,$=Object(u["a"])(k,p,g,!1,null,null,null),b=$.exports,_={components:{CardDataset:b,Filters:v},data:function(){return{hight:0}},computed:Object(c["a"])(Object(c["a"])({},Object(f["b"])({datasets:"datasets/getDatasets",project:"projects/getProject"})),{},{height:function(){return this.$store.getters["settings/height"]({deduct:"filter",padding:52,clean:!0})}}),mounted:function(){var t=this;document.addEventListener("click",(function(){t.$store.dispatch("datasets/setSelect",0),t.$store.dispatch("datasets/setSelectedIndex",null)}))},methods:{isLoaded:function(t){var e,s;return(null===(e=this.project)||void 0===e||null===(s=e.dataset)||void 0===s?void 0:s.alias)===t.alias},click:function(t,e){if(!this.isLoaded(t))return this.$store.dispatch("datasets/setSelect",t),void this.$store.dispatch("datasets/setSelectedIndex",e);this.$store.dispatch("datasets/setSelect",0),this.$store.dispatch("datasets/setSelectedIndex",null)},remove:function(t){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var a,i,n;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return a=t.name,i=t.alias,n=t.group,s.prev=1,s.next=4,e.$Modal.confirm({title:"Внимание!",content:'Вы действительно желаете удалить датасет "'.concat(a,'" ?'),width:300});case 4:return e.$store.dispatch("settings/setOverlay",!0),s.next=7,e.$store.dispatch("datasets/deleteDataset",{alias:i,group:n});case 7:e.$store.dispatch("settings/setOverlay",!1),s.next=14;break;case 10:s.prev=10,s.t0=s["catch"](1),e.$store.dispatch("settings/setOverlay",!1),console.log(s.t0);case 14:case"end":return s.stop()}}),s,null,[[1,10]])})))()}}},x=_,C=(s("e344"),Object(u["a"])(x,a,i,!1,null,"4967d41e",null));e["default"]=C.exports}}]);