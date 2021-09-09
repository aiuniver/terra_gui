<template>
  <div class="t-project">
    <div class="t-project__label">Project:</div>
    <input
      v-model="nameProject"
      v-autowidth
      class="t-project__name"
      type="text"
      maxlength="50"
      autocomplete="off"
      @keypress.enter="$event.target.blur()"
      @blur="saveProject"
      @focus="latest = nameProject"
    />
    <i class="t-icon icon-project-edit" @click="click"></i>
  </div>
</template>

<script>
export default {
  name: 't-project-name',
  data: () => ({
    latest: '',
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
    click() {
      this.$el.getElementsByTagName('input')[0].focus()
    },
    async saveProject() {
      if (this.latest === this.nameProject) return;
      this.$emit('save', this.nameProject );
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
    font-size: 1rem;
    align-items: center;
    height: 100%;
    border: 1px solid rgba(108, 120, 131, 0);
    padding: 0 5px;
    box-sizing: content-box;
    background: none;
    &:focus {
      border: 1px solid rgb(108, 120, 131);
    }
  }
  & .icon-project-edit {
    display: block;
    width: 24px;
    height: 24px;
    margin-left: 5px;
    cursor: pointer;
  }
}
</style>
