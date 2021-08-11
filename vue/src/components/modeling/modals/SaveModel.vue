<template>
  <at-modal v-model="dialog" width="680" showClose>
    <div slot="header" style="text-align: center">
      <span>Сохранить модель</span>
    </div>
    <div class="model">
      <div v-if="image" class="model__image">
        <img alt="" width="auto" height="400" :src="image || ''" />
      </div>
      <Loading v-else />
    </div>
    <div slot="footer"></div>
  </at-modal>
</template>

<script>
import Loading from '../../forms/Loading.vue';
export default {
  name: 'ModalSaveModel',
  components: {
    Loading,
  },
  props: {
    value: Boolean,
    image: String,
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
    async save() {
      await this.$store.dispatch('deploy/SendDeploy', this.model);
      this.$emit('input', false);
    },
  },
  watch: {
    dialog: {
      handler(value) {
        if (value) {
          console.log(this.refs);
        }
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.scroll-area {
  height: 350px;
}

.model {
  display: flex;
  justify-content: center;
  &__image {
    height: 400px;
  }
}
</style>