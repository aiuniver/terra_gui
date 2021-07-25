<template>
  <header class="flexbox-center-nowrap">
    <div class="project flexbox-left-nowrap">
      <a href="#" class="logo"></a>
      <div class="title flexbox-left-nowrap">
        <div class="label">Project:</div>
        <div class="name">
          <div class="value flexbox-center-nowrap">
            <span>{{nameProject}}</span><i></i>
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
        <button @click="save = false">Сохранить</button>
      </div>
    </at-modal>
    <at-modal v-model="load" width="400" >
      <div slot="header" style="text-align: center">
        <span>Загрузить проект</span>
      </div>
      
      <div slot="footer">
        
      </div>
    </at-modal>
  </header>
</template>

<script>
export default {
  name: "THeader",
  data: () => ({
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
    nameProject: 'NoName'
  }),
  computed: {
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
  },
};
</script>

<style lang="scss" scoped>
.flexbox-center-nowrap {
  padding: 0 10px;
  margin: 1px 0 0 0;
}
</style>