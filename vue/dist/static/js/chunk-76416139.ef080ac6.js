(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-76416139"],{"1be9":function(e,t,a){"use strict";a("1dcd")},"1dcd":function(e,t,a){},c753:function(e,t,a){"use strict";a.r(t);var n=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("at-modal",{attrs:{width:"400",maskClosable:!1,showClose:!1},model:{value:e.dialog,callback:function(t){e.dialog=t},expression:"dialog"}},[a("div",{attrs:{slot:"header"},slot:"header"},[a("span",[e._v("Сохранить проект")])]),a("div",{staticClass:"inner form-inline-label"},[a("t-field",{attrs:{label:"Название проекта"}},[a("d-input-text",{attrs:{type:"text",disabled:e.loading},model:{value:e.name,callback:function(t){e.name=t},expression:"name"}})],1),a("t-field",{attrs:{label:"Перезаписать"}},[a("d-checkbox",{attrs:{type:"checkbox",disabled:e.loading},model:{value:e.overwrite,callback:function(t){e.overwrite=t},expression:"overwrite"}})],1)],1),a("template",{slot:"footer"},[a("t-button",{attrs:{loading:e.loading},on:{click:function(t){return e.save({name:e.name,overwrite:e.overwrite})}}},[e._v("Сохранить")]),a("t-button",{attrs:{cancel:"",disabled:e.loading},on:{click:function(t){e.dialog=!1}}},[e._v("Отменить")])],1)],2)},i=[],s=a("1da1"),c=(a("96cf"),a("b0c0"),function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"t-field t-inline",staticStyle:{margin:"0"}},[a("label",{staticClass:"t-field__label",attrs:{for:e.parse}},[e._v(e._s(e.label))]),a("div",{class:["t-field__switch",{"t-field__switch--checked":e.checked,"t-field__switch--disabled":e.disabled}]},[a("input",{staticClass:"t-field__input",attrs:{id:e.parse,type:"checkbox",name:e.parse,disabled:e.disabled},domProps:{checked:e.checked?"checked":"",value:e.checked},on:{change:e.change}}),a("span")])])}),l=[],o=(a("caad"),a("2532"),a("56d7")),r={name:"d-checkbox",props:{label:{type:String,default:""},type:{type:String,default:"text"},value:{type:[Boolean]},name:{type:String},parse:{type:String},event:{type:Array,default:function(){return[]}},disabled:Boolean},data:function(){return{checked:null}},methods:{change:function(e){var t=e.target.checked;this.checked=t,this.$emit("input",t),this.$emit("change",{name:this.name,value:t,parse:this.parse}),o["bus"].$emit("change",{event:this.name,value:t})}},created:function(){var e=this;this.checked=this.value,this.event.length&&o["bus"].$on("change",(function(t){var a=t.event;e.event.includes(a)&&(e.checked=!1)}))},destroyed:function(){this.event.length&&(o["bus"].$off(),console.log("destroyed",this.name))}},d=r,u=(a("1be9"),a("2877")),h=Object(u["a"])(d,c,l,!1,null,"d44f4be0",null),p=h.exports,m={components:{DCheckbox:p},name:"modal-save-project",props:{value:Boolean},data:function(){return{name:"",overwrite:!1,loading:!1,disabled:!1}},computed:{dialog:{set:function(e){this.$emit("input",e)},get:function(){return this.value}}},methods:{save:function(e){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function a(){var n;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.prev=0,t.loading=!0,t.$emit("message",{message:"Сохранения проекта «".concat(e.name,"»")}),a.next=5,t.$store.dispatch("projects/saveProject",e);case 5:n=a.sent,n&&!n.error?(t.$emit("message",{message:"Проект «".concat(e.name,"» сохранен")}),t.dialog=!1,t.overwrite=!1):t.$emit("message",{error:n.error.general}),t.loading=!1,a.next=14;break;case 10:a.prev=10,a.t0=a["catch"](0),console.log(a.t0),t.loading=!1;case 14:case"end":return a.stop()}}),a,null,[[0,10]])})))()}},watch:{dialog:function(e){e&&(this.name=this.$store.getters["projects/getProject"].name)}}},v=m,f=Object(u["a"])(v,n,i,!1,null,null,null);t["default"]=f.exports}}]);