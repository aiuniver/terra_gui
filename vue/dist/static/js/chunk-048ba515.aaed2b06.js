(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-048ba515"],{4048:function(t,e,a){},d71a:function(t,e,a){"use strict";a("4048")},ec20:function(t,e,a){"use strict";a.r(e);var n=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("at-modal",{staticClass:"t-modal",attrs:{width:"300",maskClosable:!1,showClose:!1},model:{value:t.dialog,callback:function(e){t.dialog=e},expression:"dialog"}},[a("div",{attrs:{slot:"header"},slot:"header"},[a("span",[t._v("Создание проекта")])]),a("div",{staticClass:"t-modal__body"},[a("p",[t._v("Создание нового проекта удалит текущий.")])]),a("div",{staticClass:"t-modal__sub-body"},[a("span",{staticClass:"t-modal__link",on:{click:function(e){return t.$emit("start",!0)}}},[t._v("Сохранить проект")])]),a("template",{slot:"footer"},[a("t-button",{attrs:{loading:t.loading},on:{click:t.create}},[t._v("Создать")]),a("t-button",{attrs:{cancel:"",disabled:t.loading},on:{click:function(e){t.dialog=!1}}},[t._v("Отменить")])],1)],2)},o=[],s=a("1da1"),r=(a("96cf"),{name:"modal-create-project",props:{value:Boolean,loading:Boolean,disabled:Boolean},data:function(){return{}},computed:{dialog:{set:function(t){this.$emit("input",t)},get:function(){return this.value}}},methods:{create:function(){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function e(){var a;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,t.$store.dispatch("projects/createProject",{});case 3:a=e.sent,a&&!a.error&&t.$emit("message",{message:"Новый проект «NoName» создан"}),e.next=10;break;case 7:e.prev=7,e.t0=e["catch"](0),console.log(e.t0);case 10:case"end":return e.stop()}}),e,null,[[0,7]])})))()}}}),c=r,i=(a("d71a"),a("2877")),l=Object(i["a"])(c,n,o,!1,null,"0dbc3be6",null);e["default"]=l.exports}}]);