<template>
  <div class="header">
    <div class="header__left">
      <div href="#" class="header__left--logo"></div>
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
      <router-link to="/profile">
        <div class="header__right--icon">
          <i class="profile"></i>
        </div>
      </router-link>
    </div>
    <CreateProject v-model="dialogCreate" @message="message" @start="dialogSave = true" />
    <LoadProject v-model="dialogLoad" @message="message" @start="dialogSave = true" />
    <SaveProject v-model="dialogSave" @message="message" />
  </div>
</template>

<script>
import TProjectName from '../forms/TProjectName.vue';

export default {
  name: 'THeader',
  components: {
    TProjectName,
    LoadProject: () => import('./modal/LoadProject.vue'),
    CreateProject: () => import('./modal/CreateProject.vue'),
    SaveProject: () => import('./modal/SaveProject.vue'),
  },
  data: () => ({
    dialogLoad: false,
    dialogCreate: false,
    dialogSave: false,
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
    message(message) {
      this.$store.dispatch('messages/setMessage', message);
    },
    async saveNameProject(name) {
      if (this.nameProject.length > 2) {
        this.message({ message: `Изменение названия проекта на «${name}»` });
        await this.$store.dispatch('projects/saveNameProject', {
          name,
        });
        this.message({ message: `Название проекта изменено на «${name}»` });
      } else {
        this.message({ error: 'Длина не может быть < 3 сим.' });
      }
    },
    click(type) {
      if (type === 'project-new') {
        this.dialogCreate = true;
      } else if (type === 'project-save') {
        this.dialogSave = true;
      } else if (type === 'project-load') {
        this.dialogLoad = true;
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
  // position: fixed;
  // left: 0;
  // top: 0;
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
