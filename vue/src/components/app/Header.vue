<template>
  <div class="header">
    <div class="header__left">
      <a href="#" class="header__left--logo"></a>
      <TProjectName />
    </div>
    <!-- <div class="header__center">Название задачи / Название эксперимента</div> -->
    <div class="header__right">
      <div
        v-for="({ title, icon, type }, i) of items"
        :key="'menu_' + i"
        class="header__right--icon"
        @click="click(type)"
        :title="title"
      >
        <i :class="[icon]"></i>
      </div>
      <div class="header__right--icon">
        <i class="profile"></i>
      </div>
    </div>
       <at-modal
         v-model="save"
         width="400"
         :maskClosable="false"
       >
         <div slot="header" style="text-align: center">
           <span>Сохранить проект</span>
         </div>
         <div class="inner form-inline-label">
           <div class="field-form">
             <label>Название проекта</label
             ><input v-model="nameProject" type="text" />
           </div>
           <div class="field-form field-inline field-reverse">
             <label @click="checVal = !checVal">Перезаписать</label>
             <div class="checkout-switch">
               <input v-model="checVal" type="checkbox" />
               <span class="switcher"></span>
             </div>
           </div>
         </div>
         <template slot="footer">
           <button @click="saveProject">Сохранить</button>
         </template>
       </at-modal>
       <at-modal v-model="load" width="400">
         <div slot="header" style="text-align: center">
           <span>Загрузить проект</span>
         </div>

         <div slot="footer"></div>
       </at-modal>
  </div>
</template>

<script>
import TProjectName from '../forms/TProjectName.vue';

export default {
  name: 'THeader',
  components: {
    TProjectName,
  },
  data: () => ({
    checVal: false,
    clickProject: false,
    projectNameEdit: false,
    name: 'kjkjkjkj',
    items: [
      {
        title: 'Создать новый проект',
        type: 'project-new',
        icon: 'icon-project-new',
      },
      {
        title: 'Сохранить проект',
        type: 'project-save',
        icon: 'icon-project-save',
      },
      {
        title: 'Загрузить проект',
        type: 'project-load',
        icon: 'icon-project-load',
      },
    ],
    save: false,
    load: false,
  }),
  computed: {
    nameProject: {
      set(name) {
        this.$store.dispatch('projects/setProject', { name });
      },
      get() {
        return this.$store.getters['projects/getProject'].name;
      },
    },
  },
  methods: {
    async saveProject() {
      if (this.nameProject.length > 2) {
        this.$store.dispatch('messages/setMessage', {
          message: `Изменение названия проекта на «${this.nameProject}»`,
        });
        await this.$store.dispatch('projects/saveProject', {
          name: this.nameProject,
        });
        this.$store.dispatch('messages/setMessage', {
          message: `Название проекта изменено на «${this.nameProject}»`,
        });
        this.save = false;
      } else {
        this.$store.dispatch('messages/setMessage', {
          error: 'Длина не может быть < 3 сим.',
        });
      }
    },
    outside() {
      if (this.projectNameEdit) {
        this.projectNameEdit = false;
        this.nameProject = this.$refs.nameProjectSpan.innerText;
        this.saveProject();
      }
    },
    click(type) {
      console.log(type);
      if (type === 'project-new') {
        this.$Modal.confirm({
          title: 'Внимание!',
          content: 'Очистить проект?',
          width: 300
        }).then(() => {
          console.log('ok')
        }).catch(() => {
          console.log('cancel')
        })
      } else if (type === 'project-save') {
        this.save = true;
      } else if (type === 'project-load') {
        this.load = true;
      }
    },
    change(value) {
      console.log(value);
    },
    handleFocusOut(e) {
      console.log(e);
      (this.clickProject = e), this.$nextTick(() => this.$refs.project.focus());
    },
  },
};
</script>

<style lang="scss" scoped>
.header {
  background: #17212b;
  width: 100%;
  height: 52px;
  margin: 1px 0 0 0;
  padding: 0 10px;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 800;
  display: -webkit-box;
  display: -moz-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-box-direction: normal;
  -moz-box-direction: normal;
  -webkit-box-orient: horizontal;
  -moz-box-orient: horizontal;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  flex-direction: row;
  -webkit-flex-wrap: nowrap;
  -ms-flex-wrap: nowrap;
  flex-wrap: nowrap;
  -webkit-box-pack: center;
  -moz-box-pack: center;
  -webkit-justify-content: center;
  -ms-flex-pack: center;
  justify-content: center;
  -webkit-align-content: center;
  -ms-flex-line-pack: center;
  align-content: center;
  -webkit-box-align: center;
  -moz-box-align: center;
  -webkit-align-items: center;
  -ms-flex-align: center;
  align-items: center;
  &__left {
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
    display: flex;
    -webkit-box-direction: normal;
    -moz-box-direction: normal;
    -webkit-box-orient: horizontal;
    -moz-box-orient: horizontal;
    -webkit-flex-direction: row;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-flex-wrap: nowrap;
    -ms-flex-wrap: nowrap;
    flex-wrap: nowrap;
    -webkit-box-pack: start;
    -moz-box-pack: start;
    -webkit-justify-content: flex-start;
    -ms-flex-pack: start;
    justify-content: flex-start;
    -webkit-align-content: flex-start;
    -ms-flex-line-pack: start;
    align-content: flex-start;
    -webkit-box-align: center;
    -moz-box-align: center;
    -webkit-align-items: center;
    -ms-flex-align: center;
    align-items: center;
    &--logo {
      display: block;
      content: '';
      width: 28px;
      height: 28px;
      background-position: center;
      background-repeat: no-repeat;
      background-size: contain;
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIxLjQxNzUgOC4wNzA2QzIyLjIzMjggOC4wNzA2IDIyLjg5OTkgNy40MDM1NCAyMi44OTk5IDYuNTg4MjVDMjIuODk5OSA1Ljc3Mjk2IDIyLjIzMjggNS4xMDU5IDIxLjQxNzUgNS4xMDU5SDYuNTk0MDJDNS43Nzg3MiA1LjEwNTkgNS4xMTE2NiA1Ljc3Mjk2IDUuMTExNjYgNi41ODgyNUM1LjExMTY2IDcuNDAzNTQgNS43Nzg3MiA4LjA3MDYgNi41OTQwMiA4LjA3MDZIMjEuNDE3NVpNMTIuNTIzNCAyMS40MTE4QzEyLjUyMzQgMjIuMjI3MSAxMy4xOTA1IDIyLjg5NDEgMTQuMDA1OCAyMi44OTQxQzE0LjgyMTEgMjIuODk0MSAxNS40ODgxIDIyLjIyNzEgMTUuNDg4MSAyMS40MTE4VjEyLjc2NDdDMTUuNDg4MSAxMS45NDk0IDE0LjgyMTEgMTEuMjgyNCAxNC4wMDU4IDExLjI4MjRDMTMuMTkwNSAxMS4yODI0IDEyLjUyMzQgMTEuOTQ5NCAxMi41MjM0IDEyLjc2NDdWMjEuNDExOFoiIGZpbGw9IiM2NUI5RjQiLz4KPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0yNy4xNzY1IDAuODIzNTI5SDAuODIzNTI5VjI3LjE3NjVIMjcuMTc2NVYwLjgyMzUyOVpNMCAwVjI4SDI4VjBIMFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==);
    }
    &--title {
      margin: 0 0 0 15px;
      line-height: 1.375;
      display: flex;
    }
    &--label {
      color: #a7bed3;
      margin: 0 5px 0 0;
      user-select: none;
    }
    &--name {
      position: relative;
      white-space: nowrap;
      font-weight: 700;
      display: flex;
      align-items: center;
      > span {
        min-width: 20px;
        height: 100%;
        margin: 0 7px 0 0;
        cursor: text;
        user-select: none;
        white-space: nowrap;
        outline: none;
        border: 1px solid transparent;
        border-radius: 4px;
      }
      > i {
        display: block;
        width: 13px;
        height: 13px;
        background-position: center;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTMiIGhlaWdodD0iMTMiIHZpZXdCb3g9IjAgMCAxMyAxMyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTAuODc1IDkuNzgyODFWMTIuMTI2NkgzLjMxNjQxTDEwLjUxNjkgNS4yMTQwNkw4LjA3NTUyIDIuODcwMzFMMC44NzUgOS43ODI4MVpNMTIuNDA0OSAzLjQwMTU2QzEyLjY1ODkgMy4xNTc4MSAxMi42NTg5IDIuNzY0MDYgMTIuNDA0OSAyLjUyMDMxTDEwLjg4MTUgMS4wNTc4MUMxMC42Mjc2IDAuODE0MDYyIDEwLjIxNzQgMC44MTQwNjIgOS45NjM1NCAxLjA1NzgxTDguNzcyMTMgMi4yMDE1NkwxMS4yMTM1IDQuNTQ1MzFMMTIuNDA0OSAzLjQwMTU2WiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4K);
      }
    }
  }
  &__center {
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
    white-space: nowrap;
  }
  &__right {
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
    display: flex;
    -webkit-box-direction: normal;
    -moz-box-direction: normal;
    -webkit-box-orient: horizontal;
    -moz-box-orient: horizontal;
    -webkit-flex-direction: row;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-flex-wrap: nowrap;
    -ms-flex-wrap: nowrap;
    flex-wrap: nowrap;
    -webkit-box-pack: end;
    -moz-box-pack: end;
    -webkit-justify-content: flex-end;
    -ms-flex-pack: end;
    justify-content: flex-end;
    -webkit-align-content: center;
    -ms-flex-line-pack: center;
    align-content: center;
    -webkit-box-align: center;
    -moz-box-align: center;
    -webkit-align-items: center;
    -ms-flex-align: center;
    align-items: center;
    &--icon {
      margin: 0 10px 0 0;
      > i {
        background-size: 24px 24px;
        display: block;
        width: 32px;
        height: 32px;
        background-position: center;
        background-repeat: no-repeat;
        cursor: pointer;
        user-select: none;
      }
      .profile {
        background-size: 32px 32px;
        background-image: url(data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiA/PjxzdmcgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDQ4IDQ4IiB3aWR0aD0iNDgiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZmlsbD0iIzY1QjlGNCIgZD0iTTI0IDRjLTExLjA1IDAtMjAgOC45NS0yMCAyMHM4Ljk1IDIwIDIwIDIwIDIwLTguOTUgMjAtMjAtOC45NS0yMC0yMC0yMHptMCA2YzMuMzEgMCA2IDIuNjkgNiA2IDAgMy4zMi0yLjY5IDYtNiA2cy02LTIuNjgtNi02YzAtMy4zMSAyLjY5LTYgNi02em0wIDI4LjRjLTUuMDEgMC05LjQxLTIuNTYtMTItNi40NC4wNS0zLjk3IDguMDEtNi4xNiAxMi02LjE2czExLjk0IDIuMTkgMTIgNi4xNmMtMi41OSAzLjg4LTYuOTkgNi40NC0xMiA2LjQ0eiIvPjxwYXRoIGQ9Ik0wIDBoNDh2NDhoLTQ4eiIgZmlsbD0ibm9uZSIvPjwvc3ZnPg==);
      }
    }
  }
  & > * {
    -webkit-box-ordinal-group: 1;
    -moz-box-ordinal-group: 1;
    -webkit-order: 0;
    -ms-flex-order: 0;
    order: 0;
    -webkit-box-flex: 1;
    -moz-box-flex: 1;
    -webkit-flex: 1 1 auto;
    -ms-flex: 1 1 auto;
    flex: 1 1 auto;
    -webkit-align-self: auto;
    -ms-flex-item-align: auto;
    align-self: auto;
  }
}

.flexbox-center-nowrap {
  padding: 0 10px;
  margin: 1px 0 0 0;
}
.value {
  padding: 0;
  max-width: 300px;
}
.left--name_span[contenteditable='true'] {
  border: 1px solid #ffffff;
}
</style>