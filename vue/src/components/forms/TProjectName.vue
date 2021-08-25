<template>
  <div class="t-project">
    <div class="t-project__label">Project:</div>
    <input
      v-model="nameProject"
      type="text"
      class="t-project__name"
      maxlength="50"
      @blur="saveProject"
      @input="toSave = true"
      @keypress.enter="$event.target.blur()"
      v-autowidth
      @focus="latest = $store.getters['projects/getProject'].name"
    />
    <i class="t-icon icon-project-edit"></i>
  </div>
</template>

<script>
export default {
  name: 't-project-name',
  data: () => ({
    toSave: false,
    latest: ''
  }),
  computed: {
    nameProject: {
      set(name) {
        this.$store.dispatch('projects/setProject', { name });
      },
      get() {
        return this.$store.getters['projects/getProject'].name;
      },
    }
  },
  methods: {
    async saveProject() {
      if (!this.toSave || this.latest === this.nameProject) return;
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
        this.latest = this.nameProject;
      } else {
        this.$store.dispatch('messages/setMessage', {
          error: 'Длина не может быть < 3 сим.',
        });
        this.nameProject = this.latest
      }
      this.toSave = false;
    },
  }
};
</script>

<style lang="scss" scoped>
.t-project {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: flex-start;
  align-content: flex-start;
  align-items: center;
  margin-left: 10px;
  &__label {
    color: #a7bed3;
    margin: 0 5px 0 0;
    user-select: none;
  }
  &__name {
    position: relative;
    white-space: nowrap;
    font-weight: 700;
    display: flex;
    font-size: 1rem;
    align-items: center;
    height: 100%;
    border: none;
    padding: 0 5px;
    box-sizing: content-box;
    background: none;
    &:focus {
      border: 1px solid rgb(108, 120, 131);
    }
  }
  & .icon-project-edit {
    display: block;
    width: 13px;
    height: 13px;
    margin-left: 5px;
  }
}
</style>