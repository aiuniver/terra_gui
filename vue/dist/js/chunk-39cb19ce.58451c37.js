(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-39cb19ce"],{2184:function(t,e,i){"use strict";i.r(e);var n=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"header"},[i("div",{staticClass:"header-left"},[i("div",{staticClass:"header-left--logo",attrs:{href:"#"},on:{click:function(e){t.isShowNavigation=!t.isShowNavigation}}}),t.isShowNavigation?i("div",{staticClass:"header-dropdown"},[i("ul",[t._l(t.routes,(function(e){return i("li",{key:e.id,staticClass:"header-dropdown__item",on:{click:function(i){i.preventDefault(),t.selectedGeneratedId=e.id}}},[t._v(" "+t._s(e.title)+" ")])})),i("li",{staticClass:"header-dropdown__item header-dropdown__item--border"},[t._v("Тестовое поле")])],2)]):t._e(),i("svg",{attrs:{width:"24",height:"24",viewBox:"0 0 24 24",fill:"none",xmlns:"http://www.w3.org/2000/svg"},on:{click:function(e){t.isShowNavigation=!t.isShowNavigation}}},[i("path",{attrs:{d:"M12 14.5L17 9.5H7L12 14.5Z",fill:"#6C7883"}})]),i("div",{staticClass:"header-left__menu"},[i("ul",t._l(t.selectedGenerated,(function(e,n){return i("li",{key:e.title+n,staticClass:"header-left__menu-item"},[t._v(" "+t._s(e.title)+" ")])})),0)])]),i("div",{staticClass:"header-right"},[i("div",{staticClass:"header-right__line"}),t._l(t.iconRight,(function(t,e){var n=t.title,a=t.icon;return i("div",{key:"menu_"+e,staticClass:"header-right__icon",attrs:{title:n}},[i("i",{class:[a]})])})),i("router-link",{attrs:{to:"/profile"}},[i("div",{staticClass:"header-right__icon"},[i("i",{staticClass:"profile"})])])],2)])},a=[],s={name:"NewHeader",data:function(){return{iconRight:[{title:"вопрос",icon:"icon-project-ask"},{title:"уведомление",icon:"icon-project-notification"}],isShowNavigation:!1,selectedGeneratedId:1}},computed:{routes:function(){return[{id:1,title:"Данные"},{id:2,title:"Проектирование"},{id:3,title:"Обучение"},{id:4,title:"Проекты"}]},generated:function(){return{1:[{title:"Датасеты"},{title:"Создание датасета"},{title:"Разметка"},{title:"Просмотр датасета"}]}},selectedGenerated:function(){return this.generated[this.selectedGeneratedId]}}},r=s,o=(i("b0bc"),i("2877")),l=Object(o["a"])(r,n,a,!1,null,"2ed4e7df",null);e["default"]=l.exports},b0bc:function(t,e,i){"use strict";i("ec40")},ec40:function(t,e,i){}}]);