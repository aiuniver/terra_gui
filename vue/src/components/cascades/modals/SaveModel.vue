<template>
  <at-modal v-model="dialog" width="400" showClose>
    <div slot="header" style="text-align: center">
      <span>Сохранить модель</span>
    </div>
    <div class="model modal-save-model">
      <div class="model__image">
        <Loading v-if="!image" class="model__image--loading" />
        <img v-else alt="" width="auto" height="auto" :src="'data:image/png;base64,' + image || ''" />
      </div>
      <div class="model__config">
        <t-input
          :value="name"
          :label="'Название модели'"
          :type="'text'"
          :parse="'parse'"
          :name="'name'"
          :key="'name-key'"
          :error="err"
          @focus="err = ''"
          @change="name = $event.value"
        />
        <t-checkbox
          inline
          :value="overwrite"
          :label="'Перезаписать'"
          type="checkbox"
          :parse="'test'"
          :name="'overwrite'"
          :key="'overwrite'"
          @change="overwrite = $event.value"
        />
      </div>
    </div>
    <template slot="footer">
      <t-button @click.native="save" :disabled="!image">Сохранить</t-button>
      <t-button cancel @click.native="close">Отменить</t-button>
    </template>
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
  data: () => ({
    name: '',
    overwrite: false,
    err: '',
  }),
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
    change(e) {
      console.log(e);
    },
    async save() {
      if (this.image) {
        const res = await this.$store.dispatch('cascades/createModel', {
          name: this.name,
          preview: this.image,
          overwrite: this.overwrite,
        });
        if (res?.error && res?.error?.general) {
          await this.$store.dispatch('messages/setMessage', { error: `Moдель '${this.name}' уже создана` });
          this.err = `Moдель '${this.name}' уже создана`;
        } else if (res?.error?.fields) {
          this.err = res?.error?.fields?.alias;
          await this.$store.dispatch('messages/setMessage', { error: res?.error?.fields });
        } else {
          await this.$store.dispatch('messages/setMessage', { error: res?.error });
        }
        console.log(`this.err ${this.err}`);

        if (!this.err) {
          this.dialog = false;
          await this.$store.dispatch('messages/setMessage', { message: `Moдель '${this.name}' сохранена` });
        }
      }
    },

    close() {
      this.dialog = false;
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
.modal-save-model {
  flex-direction: column;

  [type='checkbox'] {
    margin-top: 10px;
  }
}
.model {
  display: flex;
  justify-content: center;

  &__image {
    height: auto;
    width: 360px;
    height: 360px;
    margin-bottom: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    &--loading {
      margin: 0 auto;
      display: block;
    }
    img {
      width: 100%;
      height: 100%;
    }
  }
}
</style>
