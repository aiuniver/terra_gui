<template>
  <at-modal v-model="dialog" class="t-modal" width="300" :maskClosable="false" :showClose="false">
    <div slot="header">
      <span>Создание проекта</span>
    </div>
    <div class="t-modal__body">
      <p>Создание нового проекта удалит текущий.</p>
    </div>
    <div class="t-modal__sub-body">
      <span class="t-modal__link" @click="$emit('start', true)">Сохранить проект</span>
    </div>
    <template slot="footer">
      <t-button @click="create" :loading="loading">Создать</t-button>
      <t-button @click="dialog = false" cancel :disabled="loading">Отменить</t-button>
    </template>
  </at-modal>
</template>

<script>
export default {
  name: 'modal-create-project',
  props: {
    value: Boolean,
    loading: Boolean,
    disabled: Boolean,
  },
  data: () => ({}),
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    async create(data) {
      try {
        const res = await this.$store.dispatch('projects/createProject', {});
        if (res && !res.error) {
          this.$emit('message', { message: `Новый проект «${data.name}» создан` });
        }
      } catch (error) {
        console.log(error);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-modal {
  &__body {
    p {
      color: #d34444;
    }
  }
  &__sub-body {
    display: flex;
    // justify-content: flex-end;
  }
  &__link {
    text-decoration: underline;
    cursor: pointer;
    font-size: 14px;
    color: #51aeff;
  }
}
</style>