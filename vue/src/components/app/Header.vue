<template>
  <div class="header">
    <div class="header__left">
      <a href="#" class="header__left--logo"></a>
      <TProjectName @save="saveNameProject" />
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
    <at-modal v-model="save" width="400" :maskClosable="false" :showClose="false">
      <div slot="header" style="text-align: center">
        <span>Сохранить модель</span>
      </div>
      <div class="inner form-inline-label">
        <div class="field-form">
          <label>Название проекта</label>
          <input v-model="nameProject" type="text" :disabled="loading" />
        </div>
        <div class="field-form field-inline field-reverse">
          <label @click="checkbox">Перезаписать</label>
          <div class="checkout-switch">
            <input v-model="checVal" type="checkbox" :disabled="loading" />
            <span class="switcher"></span>
          </div>
        </div>
      </div>
      <template slot="footer">
        <t-button @click="saveProject" :loading="loading">Сохранить</t-button>
        <t-button @click="save = false" cancel :disabled="loading">Отменить</t-button>
      </template>
    </at-modal>
    <loadProject v-model="dialog" :list="listProject" @load="loadProject" @remove="removeProject" />
  </div>
</template>

<script>
import TProjectName from '../forms/TProjectName.vue';
import loadProject from './modal/LoadProject.vue';

export default {
  name: 'THeader',
  components: {
    TProjectName,
    loadProject,
  },
  data: () => ({
    checVal: false,
    dialog: false,
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
    loading: false,
    listProject: [],
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
    checkbox() {
      if (!this.loading) {
        this.checVal = !this.checVal;
      }
    },
    message(message) {
      this.$store.dispatch('messages/setMessage', { message });
    },
    error(message) {
      this.$store.dispatch('messages/setMessage', { error: message });
    },
    async saveNameProject(name) {
      if (this.nameProject.length > 2) {
        this.message(`Изменение названия проекта на «${name}»`);
        await this.$store.dispatch('projects/saveNameProject', {
          name,
        });
        this.message(`Название проекта изменено на «${name}»`);
        this.save = false;
      } else {
        this.$store.dispatch('messages/setMessage', {
          error: 'Длина не может быть < 3 сим.',
        });
      }
    },
    async createProject() {
      try {
        await this.$Modal.confirm({
          title: 'Внимание!',
          content: 'Создание нового проекта удалит текущий. Создать новый проект?',
          width: 400,
          maskClosable: false,
          showClose: false
        });
        const res = await this.$store.dispatch('projects/createProject', {});
        if (res) {
          this.message(`Новый проект «${this.nameProject}» создан`);
        }
      } catch (error) {
        console.log(error);
      }
    },
    async loadProject(list) {
      console.log(list);
      try {
        const res = await this.$store.dispatch('projects/loadProject', {});
        console.log(res);
        if (res) {
          this.message(`Проект «${list.label}» загружен`);
          this.dialog = false;
        }
      } catch (error) {
        console.log(error);
      }
    },
    async removeProject(list) {
      console.log(list);
      try {
        const res = await this.$store.dispatch('projects/removeProject', { path: list.value });
        if (res) {
          this.message(`Проект «${list.label}» удален`);
          await this.infoProject();
        }
      } catch (error) {
        console.log(error);
      }
    },
    async infoProject() {
      try {
        const res = await this.$store.dispatch('projects/infoProject', {});
        if (res) {
          const {
            data: { projects },
          } = res;
          this.listProject = projects;
          console.log(this.listProject);
          this.dialog = true;
          // this.message(`Проект «${this.nameProject}» загружен`);
        }
      } catch (error) {
        console.log(error);
      }
    },
    async saveProject() {
      try {
        this.loading = true;
        this.message(`Сохранения проекта «${this.nameProject}»`);
        const res = await this.$store.dispatch('projects/saveProject', {
          name: this.nameProject,
          overwrite: this.checVal,
        });
        // console.log(res);
        if (res && !res.error) {
          this.message(`Проект «${this.nameProject}» сохранен`);
          this.save = false;
          this.checVal = false;
        } else {
          this.error(res.error.general);
        }
        this.loading = false;
      } catch (error) {
        console.log(error);
        this.loading = false;
      }
    },
    click(type) {
      console.log(type);
      if (type === 'project-new') {
        this.createProject();
      } else if (type === 'project-save') {
        this.save = true;
      } else if (type === 'project-load') {
        // this.load = true;
        this.infoProject();
      }
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
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: center;
  align-content: center;
  align-items: center;
  &__left {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    justify-content: flex-start;
    align-content: flex-start;
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
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    justify-content: flex-end;
    align-content: center;
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
    order: 0;
    flex: 1 1 auto;
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
