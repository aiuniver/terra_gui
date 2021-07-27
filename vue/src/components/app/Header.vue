<template>
  <header class="flexbox-center-nowrap">
    <div class="project flexbox-left-nowrap">
      <a href="#" class="logo"></a>
      <div class="title flexbox-left-nowrap">
        <div class="label">Project:</div>
        <div class="name">
          <div class="value flexbox-center-nowrap">
            <span v-show="!clickProject" @click="focusInput()">{{ nameProject }}</span>
            <input v-show="clickProject" v-model="nameProject" ref="project" type="text" @blur="blur" >            
            <i></i>
          </div>
        </div>
      </div>
    </div>
    <div class="name-experiment">Название задачи / Название эксперимента</div>
    <div class="user flexbox-right-nowrap">
      <div
        v-for="({ title, icon, type }, i) of items"
        :key="'menu_' + i"
        class="item project"
        @click="click(type)"
      >
        <div class="icon" :title="title">
          <i :class="icon"></i>
        </div>
      </div>
      <div class="item profile">
        <div class="icon"><i></i></div>
        <div class="menu">
          <div class="group">
            <ul>
              <li><span>Сменить цветовую схему</span></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
    <at-modal v-model="save" width="400">
      <div slot="header" style="text-align: center">
        <span>Сохранить проект</span>
      </div>
      <div class="inner form-inline-label">
        <div class="field-form">
          <label>Название проекта</label
          ><input v-model="nameProject" type="text" />
        </div>
        <div class="field-form field-inline field-reverse">
          <label>Перезаписать</label>
          <div class="checkout-switch">
            <input type="checkbox" />
            <span class="switcher"></span>
          </div>
        </div>
      </div>
      <div slot="footer">
        <button @click="saveProject">Сохранить</button>
      </div>
    </at-modal>
    <at-modal v-model="load" width="400">
      <div slot="header" style="text-align: center">
        <span>Загрузить проект</span>
      </div>

      <div slot="footer"></div>
    </at-modal>
  </header>
</template>

<script>
export default {
  name: "THeader",
  data: () => ({
    clickProject: false,
    items: [
      {
        title: "Создать новый проект",
        type: "project-new",
        icon: "icon-project-new",
      },
      {
        title: "Сохранить проект",
        type: "project-save",
        icon: "icon-project-save",
      },
      {
        title: "Загрузить проект",
        type: "project-load",
        icon: "icon-project-load",
      },
    ],
    save: false,
    load: false,
  }),
  computed: {
    nameProject: {
      set(name) {
        this.$store.dispatch("projects/setProject", { name });
      },
      get() {
        return this.$store.getters["projects/getProject"].name;
      },
    },
    full: {
      set(val) {
        this.$store.dispatch("datasets/setFull", val);
      },
      get() {
        return this.$store.getters["datasets/getFull"];
      },
    },
  },
  methods: {
    async saveProject() {
      await this.$store.dispatch('projects/saveProject', { name: this.nameProject })
      this.save = false
    },
    click(type) {
      console.log(type);
      if (type === "project-new") {
        this.full = !this.full;
      } else if (type === "project-save") {
        this.save = true;
      } else if (type === "project-load") {
        this.load = true;
      }
    },
    blur() {
      this.clickProject = false
      this.saveProject()
    },
    focusInput() {
      this.clickProject = true, 
      this.$nextTick(() => this.$refs.project.focus())
    }
  },
};
</script>

<style lang="scss" scoped>
.flexbox-center-nowrap {
  padding: 0 10px;
  margin: 1px 0 0 0;
}
.value {
  padding: 0;
  > input {
    width: 100px;
    height: 25px;
    margin-right: 5px;
    padding: 0 2px;
    font-size: 1rem;
    font-weight: 700;
    border: 1px solid;
    border-radius: 4px;
    transition: border-color .3s ease-in-out, opacity .3s ease-in-out;
  }
}
</style>