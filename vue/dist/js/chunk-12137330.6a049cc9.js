(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-12137330"],{c66d:function(t,e,s){"use strict";s.r(e);var a=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("main",{staticClass:"page-profile"},[s("p",{staticClass:"page-profile__title"},[t._v("Мой профиль")]),s("div",{staticClass:"page-profile__block"},[s("t-input-new",{attrs:{label:"Имя",error:t.errFirst},on:{input:function(e){t.errFirst=""}},model:{value:t.firstName,callback:function(e){t.firstName="string"===typeof e?e.trim():e},expression:"firstName"}}),s("t-input-new",{attrs:{label:"Фамилия",error:t.errLast},on:{input:function(e){t.errLast=""}},model:{value:t.lastName,callback:function(e){t.lastName="string"===typeof e?e.trim():e},expression:"lastName"}})],1),s("div",{staticClass:"page-profile__btns"},[s("t-button",{staticClass:"btn",attrs:{loading:t.isLoading,disabled:t.isLoading},on:{click:t.save}},[t._v("Сохранить")])],1),s("hr"),s("div",{staticClass:"page-profile__block"},[s("div",{staticClass:"page-profile__block--contact"},[s("p",{staticClass:"page-profile__label"},[t._v("Логин")]),s("p",{staticClass:"page-profile__text"},[t._v(t._s(t.user.login))])]),s("div",{staticClass:"page-profile__block--contact"},[s("p",{staticClass:"page-profile__label"},[t._v("E-mail")]),s("p",{staticClass:"page-profile__text"},[t._v(t._s(t.user.email))])])]),s("hr"),s("div",{staticClass:"page-profile__token"},[s("p",{staticClass:"page-profile__label"},[t._v("Token")]),s("p",{ref:"token",staticClass:"page-profile__text"},[t._v(" "+t._s(t.user.token)+" "),s("i",{staticClass:"btn-copy",on:{click:t.copy}})])]),s("hr"),t._m(0),s("transition",{attrs:{name:"slide-fade"}},[s("div",{directives:[{name:"show",rawName:"v-show",value:t.showNotice,expression:"showNotice"}],staticClass:"page-profile__notice"},[s("i",{staticClass:"notice__icon"}),s("p",[t._v(t._s(t.noticeMsg))])])])],1)},i=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"page-profile__subscription"},[s("p",{staticClass:"page-profile__label"},[t._v("Подписка действительна до 06.10.2021")])])}],n=s("1da1"),r=s("5530"),o=(s("96cf"),s("2f62")),c={name:"Profile",data:function(){return{isChanged:!1,showNotice:!1,noticeMsg:"",tId:null,errFirst:"",errLast:"",cached:null,watcher:null,isLoading:!1}},computed:Object(r["a"])(Object(r["a"])({},Object(o["b"])({user:"projects/getUser"})),{},{firstName:{set:function(t){this.$store.commit("projects/SET_USER",{first_name:t})},get:function(){return this.user.first_name}},lastName:{set:function(t){this.$store.commit("projects/SET_USER",{last_name:t})},get:function(){return this.user.last_name}}}),methods:{copy:function(){var t=window.getSelection(),e=document.createRange();e.selectNodeContents(this.$refs.token),t.removeAllRanges(),t.addRange(e),document.execCommand("copy"),t.removeAllRanges(),this.notify("Token скопирован в буфер обмена")},updateToken:function(){this.notify("Ваш token успешно обновлен")},save:function(){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function e(){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:if(t.firstName||(t.errFirst="Поле обязательно для заполнения"),t.lastName||(t.errLast="Поле обязательно для заполнения"),!t.firstName||!t.lastName){e.next=9;break}return t.isLoading=!0,e.next=6,t.$store.dispatch("profile/save",{first_name:t.firstName,last_name:t.lastName});case 6:s=e.sent,t.isLoading=!1,s.success&&t.notify("Ваши данные успешно изменены");case 9:case"end":return e.stop()}}),e)})))()},cancel:function(){var t=this;this.errFirst=this.errLast="",this.firstName=this.cached[0],this.lastName=this.cached[1],setTimeout((function(){t.isChanged=!1}),1)},notify:function(t){var e=this;clearTimeout(this.tId),this.showNotice=!1,this.noticeMsg=t,this.showNotice=!0,this.tId=setTimeout((function(){return e.showNotice=!1}),2e3)}},watch:{firstName:function(){this.isChanged=!0},lastName:function(){this.isChanged=!0}},mounted:function(){var t=this;this.watcher=this.$store.watch((function(){return t.$store.getters["projects/getUser"]}),Object(n["a"])(regeneratorRuntime.mark((function e(){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:s=Object(r["a"])({},t.user),t.cached=[s.first_name,s.last_name],t.watcher(),t.isChanged=!1;case 4:case"end":return e.stop()}}),e)}))))}},l=c,u=(s("d3dc"),s("2877")),f=Object(u["a"])(l,a,i,!1,null,"cd9a0f16",null);e["default"]=f.exports},d3dc:function(t,e,s){"use strict";s("f873")},f873:function(t,e,s){}}]);