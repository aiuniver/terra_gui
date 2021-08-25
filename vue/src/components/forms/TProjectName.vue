<template>
  <div class="t-project">
    <div class="t-project__label">Project:</div>
    <input
      v-model="nameProject"
      ref="input"
      type="text"
      class="t-project__name"
      :style="width"
      @blur="saveProject"
      @input="handleInput"
    />
    <i class="t-icon icon-project-edit"></i>
  </div>
</template>

<script>
export default {
  name: 't-project-name',
  data: () => ({
    toSave: false,
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
    width() {
      const len = this.nameProject?.length || 1
      return { width: (len < 20 ? (len * 8) : 160) + 'px' };
    },
  },
  methods: {
    handleInput() {
      this.toSave = true;
    },
    async saveProject() {
      if (!this.toSave) return;
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
        this.toSave = false;
      } else {
        this.$store.dispatch('messages/setMessage', {
          error: 'Длина не может быть < 3 сим.',
        });
      }
    },
  },
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
    align-items: center;
    height: 100%;
    max-width: 300px;
    min-width: 50px;
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