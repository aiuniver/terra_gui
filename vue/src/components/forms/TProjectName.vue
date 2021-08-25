<template>
  <div class="t-project" v-click-outside="clickShow">
    <div class="t-project__label">Project:</div>
    <input
      v-show="show"
      v-model="nameProject"
      ref="input"
      type="text"
      class="t-project__name"
      @blur="saveProject"
      @input="handleInput"
    />
    <span ref="span" class="t-project__span t-project__span--hide">{{ nameProject }}</span>
    <span v-show="!show" class="t-project__span" @click="clickShow(true)">{{ nameProject }}</span>
    <i class="t-icon icon-project-edit"></i>
  </div>
</template>

<script>
export default {
  name: 't-project-name',
  data: () => ({
    toSave: false,
    show: false,
  }),
  computed: {
    nameProject: {
      set(name) {
        if (name.length < 3) return;
        this.$store.dispatch('projects/setProject', { name });
      },
      get() {
        return this.$store.getters['projects/getProject'].name;
      },
    },
  },
  methods: {
    clickShow(value) {
      // console.log(value);
      this.show = typeof value === 'boolean';
      this.$refs.input.style.width = this.$refs?.span?.clientWidth + 10 + 'px';
      this.$nextTick(() => {
        this.$refs.input.focus();
      });
    },
    handleInput(e) {
      // console.log(this.$refs.span.clientWidth);
      var target = e.target || e.srcElement;
      target.style.width = this.$refs?.span?.clientWidth + 10 + 'px';
      this.toSave = true;
    },
    async saveProject() {
      this.show = false;
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
    z-index: 1;
    position: relative;
    white-space: nowrap;
    font-weight: 700;
    font-size: 1rem;
    height: 100%;
    width: auto;
    border: none;
    padding: 0;
    background: none;
    &:focus {
      border: 1px solid rgb(108, 120, 131);
    }
  }
  &__span {
    z-index: 1;
    height: 24px;
    font-weight: 700;
    // font-size: .875rem;
    max-width: 400px;
    min-width: 20px;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-left: 1px;
    &--hide {
      position: absolute;
      margin-left: 61px;
      opacity: 0;
      z-index: 0;
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