<template>
  <at-modal v-model="dialog" width="680" showClose>
    <div class="t-model__overlay" v-show="modelDownload">
      <LoadSpiner text="Загрузка модели" />
    </div>
    <div slot="header" class="t-model__header">
      <span>Загрузка модели</span>
      <div class="t-model__search">
        <i class="t-icon icon-search"></i>
        <t-field inline label class="t-model__field">
          <t-input-new
            v-model="search"
            ref="search"
            placeholder="Найти модель"
            type="text"
            small
            style="width: 109px"
          />
        </t-field>
      </div>
    </div>
    <div class="row at-row">
      <div class="col-16 models-list scroll-area">
        <scrollbar>
          <ul class="loaded-list">
            <li
              :class="['loaded-list__item', { 'loaded-list__item--active': selected === list.label }]"
              v-for="(list, i) in models"
              :key="`model_${i}`"
              @click="getModel(list), (selected = list.label)"
            >
              <i class="loaded-list__item--icon"></i>
              <span class="loaded-list__item--text">{{ list.label }}</span>
              <!-- <div class="loaded-list__item--empty"></div> -->
              <div class="loaded-list__item--remove" v-if="list.uid === 'custom'" @click="removeModel(list.value)">
                <i class="t-icon"></i>
              </div>
            </li>
            <li v-if="!models.length" class="loaded-list__item">
              <span class="loaded-list__item--empty">Модель "{{ search }}" не найдена</span>
            </li>
          </ul>
        </scrollbar>
      </div>
      <div class="col-8">
        <div v-if="info.name" class="model-arch">
          <div class="wrapper hidden">
            <div class="modal-arch-info">
              <div class="model-arch-info-param name">
                <span>Name:</span>
                <span>{{ info.alias ? ` ${info.name}` : '' }}</span>
              </div>
              <div class="model-arch-info-param input_shape">
                <span>Input shape:</span>
                <span>{{ info.input_shape ? ` ${info.input_shape}` : '' }}</span>
              </div>
              <!-- <div class="model-arch-info-param datatype">
                Datatype:
                <span>{{ info.name }}</span>
              </div> -->
            </div>
            <div v-if="info.image" class="model-arch__img my-5">
              <img alt="" width="100" height="200" :src="'data:image/png;base64,' + info.image || ''" />
            </div>
            <div v-if="!info.image" class="model-arch__empty"><span>Нет картинки</span></div>
            <div class="model-arch__btn">
              <t-button :disabled="!model || loading" :loading="loading" @click="download">Загрузить</t-button>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div slot="footer"></div>
  </at-modal>
</template>

<script>
import LoadSpiner from "@/components/forms/LoadSpiner";

export default {
  name: 'ModalLoadModel',
  components: {
    LoadSpiner
  },
  props: {
    value: Boolean,
  },
  data: () => ({
    lists: [],
    info: {},
    model: null,
    selected: '',
    search: '',
    loading: true,
    modelDownload: false
  }),
  mounted() {
    this.$el.getElementsByClassName('at-modal__footer')[0].remove();
  },
  created() {
    this.load();
  },
  computed: {
    models() {
      // console.log(this.lists);
      return [
        ...(this.lists[0]?.models || []).map(el => {
          return {
            ...el,
            uid: 'preset',
          };
        }),
        ...(this.lists[1]?.models || []).map(el => {
          return {
            ...el,
            uid: 'custom',
          };
        }),
      ].filter(el => el.label.match(new RegExp(this.search, 'i')));
    },
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
    async removeModel(name) {
      this.$Modal.confirm({
        title: 'Внимание!',
        content: 'Уверены, что хотите удалить эту модель?',
        width: 300,
        callback: async action => {
          if (action == 'confirm') {
            const { success } = await this.$store.dispatch('cascades/removeModel', { path: name });
            if (success) {
              this.load();
              this.selected = '';
              this.info = {};
            }
          }
        },
      });
    },
    async load() {
      const { data } = await this.$store.dispatch('cascades/info', {});
      if (data) {
        this.lists = data;
      }
    },
    async getModel(value) {
      this.loading = true;
      const { data } = await this.$store.dispatch('cascades/getModel', value);
      if (data) {
        this.info = data;
        this.model = value;
      }
      this.loading = false;
    },
    async download() {
      if (!this.loading) {
        this.loading = true;
        this.modelDownload = true;
        const { success: successValidate, data } = await this.$store.dispatch('datasets/validateDatasetOrModel', {
          model: this.model,
        });

        if (successValidate && data) {
          this.$Modal.confirm({
            title: 'Внимание!',
            content: data,
            width: 300,
            callback: async action => {
              if (action == 'confirm') {
                await this.onChoice({ reset_dataset: true });
              }
            },
          });
        } else {
          await this.onChoice();
        }
        this.$emit('input', false);
        this.loading = false;
        this.modelDownload = false;
      }
    },
    async onChoice({ reset_dataset = false } = {}) {
      await this.$store.dispatch('cascades/load', {
        model: this.model,
        reset_dataset,
      });
    },
  },
  watch: {
    dialog: {
      handler(value) {
        if (value) {
          this.$nextTick(() => {
            this.$refs.search.label();
          });
          this.load();
        }
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.t-model {
  &__overlay {
    position: fixed;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    height: 100%;
    background-color: rgb(14 22 33 / 90%);
    z-index: 801;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  &__header {
    text-align: center;
    display: flex;
    align-items: center;
    user-select: none;
  }
  &__search {
    display: flex;
    margin: 0 15px 0 auto;
    i {
      margin: 0 10px 0 0;
      width: 18px;
      opacity: 0.7;
    }
    div {
      margin: 0;
    }
  }
}
.scroll-area {
  height: 350px;
}

.loaded-list {
  padding: 3px 0;
  &__item {
    display: flex;
    align-items: center;
    transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
    cursor: pointer;
    border-radius: 0 4px 4px 0;
    padding: 10px 15px 10px 20px;
    margin-right: 15px;
    &--empty {
      line-height: 1;
      padding-left: 10px;
      font-size: 0.875rem;
      user-select: none;
      text-align: center;
      opacity: 0.7;
    }
    &--active {
      background: #2b5278;
      color: #65b9f4;
      .loaded-list__item--icon {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjE3IDBIMkMwLjkgMCAwIDAuOSAwIDJWMTZDMCAxNy4xIDAuOSAxOCAyIDE4SDE2QzE3LjEgMTggMTggMTcuMSAxOCAxNlY2LjgzQzE4IDYuMyAxNy43OSA1Ljc5IDE3LjQxIDUuNDJMMTIuNTggMC41OUMxMi4yMSAwLjIxIDExLjcgMCAxMS4xNyAwWk01IDEySDEzQzEzLjU1IDEyIDE0IDEyLjQ1IDE0IDEzQzE0IDEzLjU1IDEzLjU1IDE0IDEzIDE0SDVDNC40NSAxNCA0IDEzLjU1IDQgMTNDNCAxMi40NSA0LjQ1IDEyIDUgMTJaTTUgOEgxM0MxMy41NSA4IDE0IDguNDUgMTQgOUMxNCA5LjU1IDEzLjU1IDEwIDEzIDEwSDVDNC40NSAxMCA0IDkuNTUgNCA5QzQgOC40NSA0LjQ1IDggNSA4Wk01IDRIMTBDMTAuNTUgNCAxMSA0LjQ1IDExIDVDMTEgNS41NSAxMC41NSA2IDEwIDZINUM0LjQ1IDYgNCA1LjU1IDQgNUM0IDQuNDUgNC40NSA0IDUgNFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==');
      }
    }
    &:hover {
      background: #2b5278;
      color: #65b9f4;
      .loaded-list__item--icon {
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjE3IDBIMkMwLjkgMCAwIDAuOSAwIDJWMTZDMCAxNy4xIDAuOSAxOCAyIDE4SDE2QzE3LjEgMTggMTggMTcuMSAxOCAxNlY2LjgzQzE4IDYuMyAxNy43OSA1Ljc5IDE3LjQxIDUuNDJMMTIuNTggMC41OUMxMi4yMSAwLjIxIDExLjcgMCAxMS4xNyAwWk01IDEySDEzQzEzLjU1IDEyIDE0IDEyLjQ1IDE0IDEzQzE0IDEzLjU1IDEzLjU1IDE0IDEzIDE0SDVDNC40NSAxNCA0IDEzLjU1IDQgMTNDNCAxMi40NSA0LjQ1IDEyIDUgMTJaTTUgOEgxM0MxMy41NSA4IDE0IDguNDUgMTQgOUMxNCA5LjU1IDEzLjU1IDEwIDEzIDEwSDVDNC40NSAxMCA0IDkuNTUgNCA5QzQgOC40NSA0LjQ1IDggNSA4Wk01IDRIMTBDMTAuNTUgNCAxMSA0LjQ1IDExIDVDMTEgNS41NSAxMC41NSA2IDEwIDZINUM0LjQ1IDYgNCA1LjU1IDQgNUM0IDQuNDUgNC40NSA0IDUgNFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==');
      }
    }

    &--text {
      line-height: 1;
      padding-left: 10px;
      font-size: 0.875rem;
      user-select: none;
    }
    &--icon {
      flex: 0 0 20px;
      display: block;
      content: '';
      width: 18px;
      height: 18px;
      background-position: center;
      background-repeat: no-repeat;
      background-size: contain;
      transition: background-image 0.3s ease-in-out;
      background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTExLjE3IDBIMkMwLjkgMCAwIDAuOSAwIDJWMTZDMCAxNy4xIDAuOSAxOCAyIDE4SDE2QzE3LjEgMTggMTggMTcuMSAxOCAxNlY2LjgzQzE4IDYuMyAxNy43OSA1Ljc5IDE3LjQxIDUuNDJMMTIuNTggMC41OUMxMi4yMSAwLjIxIDExLjcgMCAxMS4xNyAwWk01IDEySDEzQzEzLjU1IDEyIDE0IDEyLjQ1IDE0IDEzQzE0IDEzLjU1IDEzLjU1IDE0IDEzIDE0SDVDNC40NSAxNCA0IDEzLjU1IDQgMTNDNCAxMi40NSA0LjQ1IDEyIDUgMTJaTTUgOEgxM0MxMy41NSA4IDE0IDguNDUgMTQgOUMxNCA5LjU1IDEzLjU1IDEwIDEzIDEwSDVDNC40NSAxMCA0IDkuNTUgNCA5QzQgOC40NSA0LjQ1IDggNSA4Wk01IDRIMTBDMTAuNTUgNCAxMSA0LjQ1IDExIDVDMTEgNS41NSAxMC41NSA2IDEwIDZINUM0LjQ1IDYgNCA1LjU1IDQgNUM0IDQuNDUgNC40NSA0IDUgNFoiIGZpbGw9IiMyQjUyNzgiLz4KPC9zdmc+Cg==');
    }
    &--empty {
      flex-grow: 1;
    }
    &--remove {
      border-radius: 2px;
      margin-right: 4px;
      padding: 2px;
      margin-left: auto;
      i {
        display: block;
        width: 18px;
        height: 18px;
        cursor: pointer;
        user-select: none;

        background-repeat: no-repeat;
        background-position: center;
        transition: background-color 0.3s ease-in-out;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTciIHZpZXdCb3g9IjAgMCAxNiAxNyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE0LjIxODUgMy4wMjIyMkgwLjk5MTk5MUMwLjcyODg5OSAzLjAyMjIyIDAuNDc2NTgyIDMuMTIxNzMgMC4yOTA1NDcgMy4yOTg4NEMwLjEwNDUxMyAzLjQ3NTk2IDAgMy43MTYxOCAwIDMuOTY2NjdDMCA0LjIxNzE1IDAuMTA0NTEzIDQuNDU3MzcgMC4yOTA1NDcgNC42MzQ0OUMwLjQ3NjU4MiA0LjgxMTYxIDAuNzI4ODk5IDQuOTExMTEgMC45OTE5OTEgNC45MTExMUgxLjMyMjY1VjEzLjc3NjNDMS4zMjQ0IDE0LjYzMDggMS42ODE3IDE1LjQ0OTggMi4zMTYzMiAxNi4wNTRDMi45NTA5NCAxNi42NTgyIDMuODExMTYgMTYuOTk4MyA0LjcwODY1IDE3SDEwLjUwMTlDMTEuMzk5NCAxNi45OTgzIDEyLjI1OTYgMTYuNjU4MiAxMi44OTQyIDE2LjA1NEMxMy41Mjg4IDE1LjQ0OTggMTMuODg2MSAxNC42MzA4IDEzLjg4NzkgMTMuNzc2M1Y0LjkxMTExSDE0LjIxODVDMTQuNDgxNiA0LjkxMTExIDE0LjczMzkgNC44MTE2MSAxNC45MiA0LjYzNDQ5QzE1LjEwNiA0LjQ1NzM3IDE1LjIxMDUgNC4yMTcxNSAxNS4yMTA1IDMuOTY2NjdDMTUuMjEwNSAzLjcxNjE4IDE1LjEwNiAzLjQ3NTk2IDE0LjkyIDMuMjk4ODRDMTQuNzMzOSAzLjEyMTczIDE0LjQ4MTYgMy4wMjIyMiAxNC4yMTg1IDMuMDIyMjJaTTExLjkwMzkgMTMuNzc2M0MxMS45MDM5IDEzLjk1MTYgMTEuODY3NiAxNC4xMjUyIDExLjc5NzIgMTQuMjg3MUMxMS43MjY3IDE0LjQ0OTEgMTEuNjIzNCAxNC41OTYyIDExLjQ5MzIgMTQuNzIwMkMxMS4zNjMxIDE0Ljg0NDEgMTEuMjA4NSAxNC45NDI0IDExLjAzODQgMTUuMDA5NUMxMC44NjgzIDE1LjA3NjYgMTAuNjg2IDE1LjExMTEgMTAuNTAxOSAxNS4xMTExSDQuNzA4NjVDNC4zMzY4MSAxNS4xMTExIDMuOTgwMjEgMTQuOTcwNSAzLjcxNzI4IDE0LjcyMDJDMy40NTQzNSAxNC40Njk4IDMuMzA2NjQgMTQuMTMwMyAzLjMwNjY0IDEzLjc3NjNWNC45MTExMUgxMS45MDM5VjEzLjc3NjNaTTMuODM1NyAwLjk0NDQ0NEMzLjgzNTcgMC42OTM5NjIgMy45NDAyMSAwLjQ1MzczOSA0LjEyNjI1IDAuMjc2NjIxQzQuMzEyMjggMC4wOTk1MDM3IDQuNTY0NiAwIDQuODI3NjkgMEgxMC4zODI4QzEwLjY0NTkgMCAxMC44OTgyIDAuMDk5NTAzNyAxMS4wODQzIDAuMjc2NjIxQzExLjI3MDMgMC40NTM3MzkgMTEuMzc0OCAwLjY5Mzk2MiAxMS4zNzQ4IDAuOTQ0NDQ0QzExLjM3NDggMS4xOTQ5MyAxMS4yNzAzIDEuNDM1MTUgMTEuMDg0MyAxLjYxMjI3QzEwLjg5ODIgMS43ODkzOSAxMC42NDU5IDEuODg4ODkgMTAuMzgyOCAxLjg4ODg5SDQuODI3NjlDNC41NjQ2IDEuODg4ODkgNC4zMTIyOCAxLjc4OTM5IDQuMTI2MjUgMS42MTIyN0MzLjk0MDIxIDEuNDM1MTUgMy44MzU3IDEuMTk0OTMgMy44MzU3IDAuOTQ0NDQ0Wk00LjY5NTQyIDEyLjUyOTZWNy40OTI1OUM0LjY5NTQyIDcuMjQyMTEgNC43OTk5NCA3LjAwMTg5IDQuOTg1OTcgNi44MjQ3N0M1LjE3MjAxIDYuNjQ3NjUgNS40MjQzMiA2LjU0ODE1IDUuNjg3NDEgNi41NDgxNUM1Ljk1MDUxIDYuNTQ4MTUgNi4yMDI4MiA2LjY0NzY1IDYuMzg4ODYgNi44MjQ3N0M2LjU3NDg5IDcuMDAxODkgNi42Nzk0MSA3LjI0MjExIDYuNjc5NDEgNy40OTI1OVYxMi41Mjk2QzYuNjc5NDEgMTIuNzgwMSA2LjU3NDg5IDEzLjAyMDMgNi4zODg4NiAxMy4xOTc1QzYuMjAyODIgMTMuMzc0NiA1Ljk1MDUxIDEzLjQ3NDEgNS42ODc0MSAxMy40NzQxQzUuNDI0MzIgMTMuNDc0MSA1LjE3MjAxIDEzLjM3NDYgNC45ODU5NyAxMy4xOTc1QzQuNzk5OTQgMTMuMDIwMyA0LjY5NTQyIDEyLjc4MDEgNC42OTU0MiAxMi41Mjk2Wk04LjUzMTEyIDEyLjUyOTZWNy40OTI1OUM4LjUzMTEyIDcuMjQyMTEgOC42MzU2MyA3LjAwMTg5IDguODIxNjcgNi44MjQ3N0M5LjAwNzcgNi42NDc2NSA5LjI2MDAyIDYuNTQ4MTUgOS41MjMxMSA2LjU0ODE1QzkuNzg2MiA2LjU0ODE1IDEwLjAzODUgNi42NDc2NSAxMC4yMjQ2IDYuODI0NzdDMTAuNDEwNiA3LjAwMTg5IDEwLjUxNTEgNy4yNDIxMSAxMC41MTUxIDcuNDkyNTlWMTIuNTI5NkMxMC41MTUxIDEyLjc4MDEgMTAuNDEwNiAxMy4wMjAzIDEwLjIyNDYgMTMuMTk3NUMxMC4wMzg1IDEzLjM3NDYgOS43ODYyIDEzLjQ3NDEgOS41MjMxMSAxMy40NzQxQzkuMjYwMDIgMTMuNDc0MSA5LjAwNzcgMTMuMzc0NiA4LjgyMTY3IDEzLjE5NzVDOC42MzU2MyAxMy4wMjAzIDguNTMxMTIgMTIuNzgwMSA4LjUzMTEyIDEyLjUyOTZaIiBmaWxsPSIjNjVCOUY0Ii8+Cjwvc3ZnPg==');
      }
      &:hover {
        background: rgba(255, 255, 255, 0.2);
      }
    }
  }
}

.model-arch {
  &__img {
    border: 1px solid #6c7883;
    border-radius: 4px;
    background-color: #242f3d;
    img {
      width: 100%;
      height: 198px;
      object-fit: contain;
    }
  }
  &__empty {
    display: flex;
    justify-content: center;
    align-items: center;
    color: #6c7883;
    width: 100%;
    height: 200px;
    border: 1px solid #6c7883;
    border-radius: 4px;
    background-repeat: no-repeat;
    background-position: center;
    margin: 20px 0;
    background-color: #242f3d;
    background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTc4IiBoZWlnaHQ9IjE0NiIgdmlld0JveD0iMCAwIDE3OCAxNDYiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxnIG9wYWNpdHk9IjAuNCI+CjxwYXRoIGQ9Ik03NS4wNzU1IDMuMDIxMDVDNzUuMDc1NSAxLjkwNDg2IDc1Ljk3OCAxIDc3LjA5MTIgMUgxMDUuMjk4QzEwNi40MTIgMSAxMDcuMzE0IDEuOTA0ODYgMTA3LjMxNCAzLjAyMTA1VjE2LjY2MzJDMTA3LjMxNCAxNy43Nzk0IDEwNi40MTIgMTguNjg0MiAxMDUuMjk4IDE4LjY4NDJINzcuMDkxMkM3NS45NzggMTguNjg0MiA3NS4wNzU1IDE3Ljc3OTQgNzUuMDc1NSAxNi42NjMyVjMuMDIxMDVaIiBmaWxsPSIjNDk0MzQzIi8+CjxwYXRoIGQ9Ik00OS4zNzU5IDEyOS4zMzdDNDkuMzc1OSAxMjguMjIxIDUwLjI3ODMgMTI3LjMxNiA1MS4zOTE1IDEyNy4zMTZINzYuODY2NkM3Ny45Nzk4IDEyNy4zMTYgNzguODgyMiAxMjguMjIxIDc4Ljg4MjIgMTI5LjMzN1YxNDIuOTc5Qzc4Ljg4MjIgMTQ0LjA5NSA3Ny45Nzk4IDE0NSA3Ni44NjY2IDE0NUg1MS4zOTE1QzUwLjI3ODMgMTQ1IDQ5LjM3NTkgMTQ0LjA5NSA0OS4zNzU5IDE0Mi45NzlWMTI5LjMzN1oiIGZpbGw9IiM0OTQzNDMiLz4KPHBhdGggZD0iTTc5LjEwNjkgMTA3LjEwNUM3OS4xMDY5IDEwNS45ODkgODAuMDA5MyAxMDUuMDg0IDgxLjEyMjUgMTA1LjA4NEgxMDYuNTk4QzEwNy43MTEgMTA1LjA4NCAxMDguNjEzIDEwNS45ODkgMTA4LjYxMyAxMDcuMTA1VjEyMC43NDdDMTA4LjYxMyAxMjEuODY0IDEwNy43MTEgMTIyLjc2OCAxMDYuNTk4IDEyMi43NjhIODEuMTIyNUM4MC4wMDkzIDEyMi43NjggNzkuMTA2OSAxMjEuODY0IDc5LjEwNjkgMTIwLjc0N1YxMDcuMTA1WiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNNjIuOTgxNiAzMS44MjExQzYyLjk4MTYgMzAuNzA0OSA2My44ODQgMjkuOCA2NC45OTcyIDI5LjhIODUuNTU0NkM4Ni42Njc4IDI5LjggODcuNTcwMiAzMC43MDQ5IDg3LjU3MDIgMzEuODIxMVYzNy44ODQyQzg3LjU3MDIgMzkuMDAwNCA4Ni42Njc4IDM5LjkwNTMgODUuNTU0NiAzOS45MDUzSDY0Ljk5NzJDNjMuODg0IDM5LjkwNTMgNjIuOTgxNiAzOS4wMDA0IDYyLjk4MTYgMzcuODg0MlYzMS44MjExWiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNNjIuOTgxNiA1My41NDc0QzYyLjk4MTYgNTIuNDMxMiA2My44ODQgNTEuNTI2MyA2NC45OTcyIDUxLjUyNjNIODUuNTU0NkM4Ni42Njc4IDUxLjUyNjMgODcuNTcwMiA1Mi40MzEyIDg3LjU3MDIgNTMuNTQ3NFY1OS42MTA1Qzg3LjU3MDIgNjAuNzI2NyA4Ni42Njc4IDYxLjYzMTYgODUuNTU0NiA2MS42MzE2SDY0Ljk5NzJDNjMuODg0IDYxLjYzMTYgNjIuOTgxNiA2MC43MjY3IDYyLjk4MTYgNTkuNjEwNVY1My41NDc0WiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNMTQ2LjEyOCAzMi4zMjYzQzE0Ni4xMjggMzEuMjEwMSAxNDcuMDMgMzAuMzA1MyAxNDguMTQzIDMwLjMwNTNIMTY4LjcwMUMxNjkuODE0IDMwLjMwNTMgMTcwLjcxNiAzMS4yMTAxIDE3MC43MTYgMzIuMzI2M1YzOC4zODk1QzE3MC43MTYgMzkuNTA1NyAxNjkuODE0IDQwLjQxMDUgMTY4LjcwMSA0MC40MTA1SDE0OC4xNDNDMTQ3LjAzIDQwLjQxMDUgMTQ2LjEyOCAzOS41MDU3IDE0Ni4xMjggMzguMzg5NVYzMi4zMjYzWiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNNjMuOTg5NCA2OS43MTU4QzYzLjk4OTQgNjguNTk5NiA2NC44OTE4IDY3LjY5NDcgNjYuMDA1MSA2Ny42OTQ3SDg2LjU2MjRDODcuNjc1NiA2Ny42OTQ3IDg4LjU3OCA2OC41OTk2IDg4LjU3OCA2OS43MTU4Vjc1Ljc3ODlDODguNTc4IDc2Ljg5NTEgODcuNjc1NiA3Ny44IDg2LjU2MjQgNzcuOEg2Ni4wMDUxQzY0Ljg5MTggNzcuOCA2My45ODk0IDc2Ljg5NTEgNjMuOTg5NCA3NS43Nzg5VjY5LjcxNThaIiBmaWxsPSIjNDk0MzQzIi8+CjxwYXRoIGQ9Ik02My45ODk0IDg1LjM3ODlDNjMuOTg5NCA4NC4yNjI3IDY0Ljg5MTggODMuMzU3OSA2Ni4wMDUxIDgzLjM1NzlIODYuNTYyNEM4Ny42NzU2IDgzLjM1NzkgODguNTc4IDg0LjI2MjcgODguNTc4IDg1LjM3ODlWOTEuNDQyMUM4OC41NzggOTIuNTU4MyA4Ny42NzU2IDkzLjQ2MzIgODYuNTYyNCA5My40NjMySDY2LjAwNTFDNjQuODkxOCA5My40NjMyIDYzLjk4OTQgOTIuNTU4MyA2My45ODk0IDkxLjQ0MjFWODUuMzc4OVoiIGZpbGw9IiM0OTQzNDMiLz4KPHBhdGggZD0iTTMzLjc1NDUgNjkuNzE1OEMzMy43NTQ1IDY4LjU5OTYgMzQuNjU2OSA2Ny42OTQ3IDM1Ljc3MDIgNjcuNjk0N0g1Ni4zMjc1QzU3LjQ0MDcgNjcuNjk0NyA1OC4zNDMxIDY4LjU5OTYgNTguMzQzMSA2OS43MTU4Vjc1Ljc3ODlDNTguMzQzMSA3Ni44OTUxIDU3LjQ0MDcgNzcuOCA1Ni4zMjc1IDc3LjhIMzUuNzcwMkMzNC42NTY5IDc3LjggMzMuNzU0NSA3Ni44OTUxIDMzLjc1NDUgNzUuNzc4OVY2OS43MTU4WiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNMSA2OS43MTU4QzEgNjguNTk5NiAxLjkwMjQ0IDY3LjY5NDcgMy4wMTU2NiA2Ny42OTQ3SDIzLjU3M0MyNC42ODYyIDY3LjY5NDcgMjUuNTg4NiA2OC41OTk2IDI1LjU4ODYgNjkuNzE1OFY3NS43Nzg5QzI1LjU4ODYgNzYuODk1MSAyNC42ODYyIDc3LjggMjMuNTczIDc3LjhIMy4wMTU2NkMxLjkwMjQ0IDc3LjggMSA3Ni44OTUxIDEgNzUuNzc4OVY2OS43MTU4WiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNMzMuNzU0NSA4NS4zNzg5QzMzLjc1NDUgODQuMjYyNyAzNC42NTY5IDgzLjM1NzkgMzUuNzcwMiA4My4zNTc5SDU2LjMyNzVDNTcuNDQwNyA4My4zNTc5IDU4LjM0MzEgODQuMjYyNyA1OC4zNDMxIDg1LjM3ODlWOTEuNDQyMUM1OC4zNDMxIDkyLjU1ODMgNTcuNDQwNyA5My40NjMyIDU2LjMyNzUgOTMuNDYzMkgzNS43NzAyQzM0LjY1NjkgOTMuNDYzMiAzMy43NTQ1IDkyLjU1ODMgMzMuNzU0NSA5MS40NDIxVjg1LjM3ODlaIiBmaWxsPSIjNDk0MzQzIi8+CjxwYXRoIGQ9Ik0zMy43NTQ1IDEwMi4wNTNDMzMuNzU0NSAxMDAuOTM2IDM0LjY1NjkgMTAwLjAzMiAzNS43NzAyIDEwMC4wMzJINTYuMzI3NUM1Ny40NDA3IDEwMC4wMzIgNTguMzQzMSAxMDAuOTM2IDU4LjM0MzEgMTAyLjA1M1YxMDguMTE2QzU4LjM0MzEgMTA5LjIzMiA1Ny40NDA3IDExMC4xMzcgNTYuMzI3NSAxMTAuMTM3SDM1Ljc3MDJDMzQuNjU2OSAxMTAuMTM3IDMzLjc1NDUgMTA5LjIzMiAzMy43NTQ1IDEwOC4xMTZWMTAyLjA1M1oiIGZpbGw9IiM0OTQzNDMiLz4KPHBhdGggZD0iTTMyLjc0NjcgNTMuNTQ3NEMzMi43NDY3IDUyLjQzMTIgMzMuNjQ5MSA1MS41MjYzIDM0Ljc2MjMgNTEuNTI2M0g1NS4zMTk2QzU2LjQzMjkgNTEuNTI2MyA1Ny4zMzUzIDUyLjQzMTIgNTcuMzM1MyA1My41NDc0VjU5LjYxMDVDNTcuMzM1MyA2MC43MjY3IDU2LjQzMjkgNjEuNjMxNiA1NS4zMTk2IDYxLjYzMTZIMzQuNzYyM0MzMy42NDkxIDYxLjYzMTYgMzIuNzQ2NyA2MC43MjY3IDMyLjc0NjcgNTkuNjEwNVY1My41NDc0WiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNMTA0LjgwNyA2MC42MjExQzEwNC44MDcgNTkuNTA0OSAxMDUuNzA5IDU4LjYgMTA2LjgyMiA1OC42SDEzMy42NjNDMTM0Ljc3NiA1OC42IDEzNS42NzkgNTkuNTA0OSAxMzUuNjc5IDYwLjYyMTFWNjYuNjg0MkMxMzUuNjc5IDY3LjgwMDQgMTM0Ljc3NiA2OC43MDUzIDEzMy42NjMgNjguNzA1M0gxMDYuODIyQzEwNS43MDkgNjguNzA1MyAxMDQuODA3IDY3LjgwMDQgMTA0LjgwNyA2Ni42ODQyVjYwLjYyMTFaIiBmaWxsPSIjNDk0MzQzIi8+CjxwYXRoIGQ9Ik0xMDUuMzEgNzQuMjYzMkMxMDUuMzEgNzMuMTQ3IDEwNi4yMTMgNzIuMjQyMSAxMDcuMzI2IDcyLjI0MjFIMTM0LjE2N0MxMzUuMjggNzIuMjQyMSAxMzYuMTgzIDczLjE0NyAxMzYuMTgzIDc0LjI2MzJWODAuMzI2M0MxMzYuMTgzIDgxLjQ0MjUgMTM1LjI4IDgyLjM0NzQgMTM0LjE2NyA4Mi4zNDc0SDEwNy4zMjZDMTA2LjIxMyA4Mi4zNDc0IDEwNS4zMSA4MS40NDI1IDEwNS4zMSA4MC4zMjYzVjc0LjI2MzJaIiBmaWxsPSIjNDk0MzQzIi8+CjxwYXRoIGQ9Ik0xMTkuOTI0IDkwLjQzMTZDMTE5LjkyNCA4OS4zMTU0IDEyMC44MjYgODguNDEwNSAxMjEuOTQgODguNDEwNUgxNDguNzgxQzE0OS44OTQgODguNDEwNSAxNTAuNzk2IDg5LjMxNTQgMTUwLjc5NiA5MC40MzE2VjEwNC4wNzRDMTUwLjc5NiAxMDUuMTkgMTQ5Ljg5NCAxMDYuMDk1IDE0OC43ODEgMTA2LjA5NUgxMjEuOTRDMTIwLjgyNiAxMDYuMDk1IDExOS45MjQgMTA1LjE5IDExOS45MjQgMTA0LjA3NFY5MC40MzE2WiIgZmlsbD0iIzQ5NDM0MyIvPgo8cGF0aCBkPSJNMTQ2LjEyOCA0OC40OTQ3QzE0Ni4xMjggNDcuMzc4NSAxNDcuMDMgNDYuNDczNyAxNDguMTQzIDQ2LjQ3MzdIMTc0Ljk4NEMxNzYuMDk4IDQ2LjQ3MzcgMTc3IDQ3LjM3ODUgMTc3IDQ4LjQ5NDdWNjIuMTM2OEMxNzcgNjMuMjUzIDE3Ni4wOTggNjQuMTU3OSAxNzQuOTg0IDY0LjE1NzlIMTQ4LjE0M0MxNDcuMDMgNjQuMTU3OSAxNDYuMTI4IDYzLjI1MyAxNDYuMTI4IDYyLjEzNjhWNDguNDk0N1oiIGZpbGw9IiM0OTQzNDMiLz4KPHBhdGggZD0iTTkxLjIwMDggMTguNjg0MkM4Ny44NzIgMjUuMTkxIDgxLjY0NjcgMjkuOCA3NS4wNzU1IDI5LjhNOTEuMjAwOCAxOC42ODQyQzk3LjEzMDMgNDIuMDQ5NSAxMDguMjE5IDU4LjYgMTE5LjkyNCA1OC42TTkxLjIwMDggMTguNjg0MkMxMTkuMTI4IDE4LjY4NDIgMTQ1LjU4NiAyMy4yOTMyIDE1OS43MzMgMjkuOE02NC45OTcyIDM4LjM4OTVDNjEuMjUyMyA0Ni4wNzkzIDU0LjI0ODggNTEuNTI2MyA0Ni44NTYzIDUxLjUyNjNNNzYuNTg3MyA0MC40MTA1VjUxLjUyNjNNNzcuMDkxMiA2MS42MzE2Qzc2Ljk4NzIgNjUuMTgwNyA3Ni43OTI2IDY3LjY5NDcgNzYuNTg3MyA2Ny42OTQ3TTE2MC4yMzcgNDAuNDEwNUMxNjAuMTMzIDQzLjk1OTcgMTU5LjkzOSA0Ni40NzM3IDE1OS43MzMgNDYuNDczN003Ny4wOTEyIDc3LjI5NDdDNzYuOTg3MiA4MC44NDM5IDc2Ljc5MjYgODMuMzU3OSA3Ni41ODczIDgzLjM1NzlNMTIyLjk0NyA4Mi4zNDc0QzEyNS4xMzIgODUuODk2NSAxMjkuMjE3IDg4LjQxMDUgMTMzLjUzIDg4LjQxMDVNNDMuODMyOCA2MS42MzE2QzQ0LjA0MDggNjUuMTgwNyA0NC40Mjk5IDY3LjY5NDcgNDQuODQwNiA2Ny42OTQ3TTMyLjc0NjcgNjEuNjMxNkMyOC40ODE2IDY1LjE4MDcgMjAuNTA1NCA2Ny42OTQ3IDEyLjA4NjEgNjcuNjk0N000My44MzI4IDc3LjI5NDdDNDQuMDQwOCA4MC44NDM5IDQ0LjQyOTkgODMuMzU3OSA0NC44NDA2IDgzLjM1NzlNNDUuODQ4NSA5My40NjMyQzQ2LjA1NjUgOTcuMDEyMyA0Ni40NDU2IDk5LjUyNjMgNDYuODU2MyA5OS41MjYzTTQ2LjM1MjQgMTEwLjEzN0M1MC4yMDEzIDEyMC4xOTMgNTcuMzk5MyAxMjcuMzE2IDY0Ljk5NzIgMTI3LjMxNk03Ni4wODM0IDkzLjQ2MzJDNzkuNzI0MyAxMDAuMjY2IDg2LjUzMzIgMTA1LjA4NCA5My43MjA0IDEwNS4wODRNNzcuMDkxMiAxOC42ODQySDEwNS4yOThDMTA2LjQxMiAxOC42ODQyIDEwNy4zMTQgMTcuNzc5NCAxMDcuMzE0IDE2LjY2MzJWMy4wMjEwNUMxMDcuMzE0IDEuOTA0ODYgMTA2LjQxMiAxIDEwNS4yOTggMUg3Ny4wOTEyQzc1Ljk3OCAxIDc1LjA3NTUgMS45MDQ4NiA3NS4wNzU1IDMuMDIxMDVWMTYuNjYzMkM3NS4wNzU1IDE3Ljc3OTQgNzUuOTc4IDE4LjY4NDIgNzcuMDkxMiAxOC42ODQyWk01MS4zOTE1IDE0NUg3Ni44NjY2Qzc3Ljk3OTggMTQ1IDc4Ljg4MjIgMTQ0LjA5NSA3OC44ODIyIDE0Mi45NzlWMTI5LjMzN0M3OC44ODIyIDEyOC4yMjEgNzcuOTc5OCAxMjcuMzE2IDc2Ljg2NjYgMTI3LjMxNkg1MS4zOTE1QzUwLjI3ODMgMTI3LjMxNiA0OS4zNzU5IDEyOC4yMjEgNDkuMzc1OSAxMjkuMzM3VjE0Mi45NzlDNDkuMzc1OSAxNDQuMDk1IDUwLjI3ODMgMTQ1IDUxLjM5MTUgMTQ1Wk04MS4xMjI1IDEyMi43NjhIMTA2LjU5OEMxMDcuNzExIDEyMi43NjggMTA4LjYxMyAxMjEuODY0IDEwOC42MTMgMTIwLjc0N1YxMDcuMTA1QzEwOC42MTMgMTA1Ljk4OSAxMDcuNzExIDEwNS4wODQgMTA2LjU5OCAxMDUuMDg0SDgxLjEyMjVDODAuMDA5MyAxMDUuMDg0IDc5LjEwNjkgMTA1Ljk4OSA3OS4xMDY5IDEwNy4xMDVWMTIwLjc0N0M3OS4xMDY5IDEyMS44NjQgODAuMDA5MyAxMjIuNzY4IDgxLjEyMjUgMTIyLjc2OFpNNjQuOTk3MiAzOS45MDUzSDg1LjU1NDZDODYuNjY3OCAzOS45MDUzIDg3LjU3MDIgMzkuMDAwNCA4Ny41NzAyIDM3Ljg4NDJWMzEuODIxMUM4Ny41NzAyIDMwLjcwNDkgODYuNjY3OCAyOS44IDg1LjU1NDYgMjkuOEg2NC45OTcyQzYzLjg4NCAyOS44IDYyLjk4MTYgMzAuNzA0OSA2Mi45ODE2IDMxLjgyMTFWMzcuODg0MkM2Mi45ODE2IDM5LjAwMDQgNjMuODg0IDM5LjkwNTMgNjQuOTk3MiAzOS45MDUzWk02NC45OTcyIDYxLjYzMTZIODUuNTU0NkM4Ni42Njc4IDYxLjYzMTYgODcuNTcwMiA2MC43MjY3IDg3LjU3MDIgNTkuNjEwNVY1My41NDc0Qzg3LjU3MDIgNTIuNDMxMiA4Ni42Njc4IDUxLjUyNjMgODUuNTU0NiA1MS41MjYzSDY0Ljk5NzJDNjMuODg0IDUxLjUyNjMgNjIuOTgxNiA1Mi40MzEyIDYyLjk4MTYgNTMuNTQ3NFY1OS42MTA1QzYyLjk4MTYgNjAuNzI2NyA2My44ODQgNjEuNjMxNiA2NC45OTcyIDYxLjYzMTZaTTE0OC4xNDMgNDAuNDEwNUgxNjguNzAxQzE2OS44MTQgNDAuNDEwNSAxNzAuNzE2IDM5LjUwNTcgMTcwLjcxNiAzOC4zODk1VjMyLjMyNjNDMTcwLjcxNiAzMS4yMTAxIDE2OS44MTQgMzAuMzA1MyAxNjguNzAxIDMwLjMwNTNIMTQ4LjE0M0MxNDcuMDMgMzAuMzA1MyAxNDYuMTI4IDMxLjIxMDEgMTQ2LjEyOCAzMi4zMjYzVjM4LjM4OTVDMTQ2LjEyOCAzOS41MDU3IDE0Ny4wMyA0MC40MTA1IDE0OC4xNDMgNDAuNDEwNVpNNjYuMDA1MSA3Ny44SDg2LjU2MjRDODcuNjc1NiA3Ny44IDg4LjU3OCA3Ni44OTUxIDg4LjU3OCA3NS43Nzg5VjY5LjcxNThDODguNTc4IDY4LjU5OTYgODcuNjc1NiA2Ny42OTQ3IDg2LjU2MjQgNjcuNjk0N0g2Ni4wMDUxQzY0Ljg5MTggNjcuNjk0NyA2My45ODk0IDY4LjU5OTYgNjMuOTg5NCA2OS43MTU4Vjc1Ljc3ODlDNjMuOTg5NCA3Ni44OTUxIDY0Ljg5MTggNzcuOCA2Ni4wMDUxIDc3LjhaTTY2LjAwNTEgOTMuNDYzMkg4Ni41NjI0Qzg3LjY3NTYgOTMuNDYzMiA4OC41NzggOTIuNTU4MyA4OC41NzggOTEuNDQyMVY4NS4zNzg5Qzg4LjU3OCA4NC4yNjI3IDg3LjY3NTYgODMuMzU3OSA4Ni41NjI0IDgzLjM1NzlINjYuMDA1MUM2NC44OTE4IDgzLjM1NzkgNjMuOTg5NCA4NC4yNjI3IDYzLjk4OTQgODUuMzc4OVY5MS40NDIxQzYzLjk4OTQgOTIuNTU4MyA2NC44OTE4IDkzLjQ2MzIgNjYuMDA1MSA5My40NjMyWk0zNS43NzAyIDc3LjhINTYuMzI3NUM1Ny40NDA3IDc3LjggNTguMzQzMSA3Ni44OTUxIDU4LjM0MzEgNzUuNzc4OVY2OS43MTU4QzU4LjM0MzEgNjguNTk5NiA1Ny40NDA3IDY3LjY5NDcgNTYuMzI3NSA2Ny42OTQ3SDM1Ljc3MDJDMzQuNjU2OSA2Ny42OTQ3IDMzLjc1NDUgNjguNTk5NiAzMy43NTQ1IDY5LjcxNThWNzUuNzc4OUMzMy43NTQ1IDc2Ljg5NTEgMzQuNjU2OSA3Ny44IDM1Ljc3MDIgNzcuOFpNMy4wMTU2NiA3Ny44SDIzLjU3M0MyNC42ODYyIDc3LjggMjUuNTg4NiA3Ni44OTUxIDI1LjU4ODYgNzUuNzc4OVY2OS43MTU4QzI1LjU4ODYgNjguNTk5NiAyNC42ODYyIDY3LjY5NDcgMjMuNTczIDY3LjY5NDdIMy4wMTU2NkMxLjkwMjQ0IDY3LjY5NDcgMSA2OC41OTk2IDEgNjkuNzE1OFY3NS43Nzg5QzEgNzYuODk1MSAxLjkwMjQ0IDc3LjggMy4wMTU2NiA3Ny44Wk0zNS43NzAyIDkzLjQ2MzJINTYuMzI3NUM1Ny40NDA3IDkzLjQ2MzIgNTguMzQzMSA5Mi41NTgzIDU4LjM0MzEgOTEuNDQyMVY4NS4zNzg5QzU4LjM0MzEgODQuMjYyNyA1Ny40NDA3IDgzLjM1NzkgNTYuMzI3NSA4My4zNTc5SDM1Ljc3MDJDMzQuNjU2OSA4My4zNTc5IDMzLjc1NDUgODQuMjYyNyAzMy43NTQ1IDg1LjM3ODlWOTEuNDQyMUMzMy43NTQ1IDkyLjU1ODMgMzQuNjU2OSA5My40NjMyIDM1Ljc3MDIgOTMuNDYzMlpNMzUuNzcwMiAxMTAuMTM3SDU2LjMyNzVDNTcuNDQwNyAxMTAuMTM3IDU4LjM0MzEgMTA5LjIzMiA1OC4zNDMxIDEwOC4xMTZWMTAyLjA1M0M1OC4zNDMxIDEwMC45MzYgNTcuNDQwNyAxMDAuMDMyIDU2LjMyNzUgMTAwLjAzMkgzNS43NzAyQzM0LjY1NjkgMTAwLjAzMiAzMy43NTQ1IDEwMC45MzYgMzMuNzU0NSAxMDIuMDUzVjEwOC4xMTZDMzMuNzU0NSAxMDkuMjMyIDM0LjY1NjkgMTEwLjEzNyAzNS43NzAyIDExMC4xMzdaTTM0Ljc2MjMgNjEuNjMxNkg1NS4zMTk2QzU2LjQzMjkgNjEuNjMxNiA1Ny4zMzUzIDYwLjcyNjcgNTcuMzM1MyA1OS42MTA1VjUzLjU0NzRDNTcuMzM1MyA1Mi40MzEyIDU2LjQzMjkgNTEuNTI2MyA1NS4zMTk2IDUxLjUyNjNIMzQuNzYyM0MzMy42NDkxIDUxLjUyNjMgMzIuNzQ2NyA1Mi40MzEyIDMyLjc0NjcgNTMuNTQ3NFY1OS42MTA1QzMyLjc0NjcgNjAuNzI2NyAzMy42NDkxIDYxLjYzMTYgMzQuNzYyMyA2MS42MzE2Wk0xMDYuODIyIDY4LjcwNTNIMTMzLjY2M0MxMzQuNzc2IDY4LjcwNTMgMTM1LjY3OSA2Ny44MDA0IDEzNS42NzkgNjYuNjg0MlY2MC42MjExQzEzNS42NzkgNTkuNTA0OSAxMzQuNzc2IDU4LjYgMTMzLjY2MyA1OC42SDEwNi44MjJDMTA1LjcwOSA1OC42IDEwNC44MDcgNTkuNTA0OSAxMDQuODA3IDYwLjYyMTFWNjYuNjg0MkMxMDQuODA3IDY3LjgwMDQgMTA1LjcwOSA2OC43MDUzIDEwNi44MjIgNjguNzA1M1pNMTA3LjMyNiA4Mi4zNDc0SDEzNC4xNjdDMTM1LjI4IDgyLjM0NzQgMTM2LjE4MyA4MS40NDI1IDEzNi4xODMgODAuMzI2M1Y3NC4yNjMyQzEzNi4xODMgNzMuMTQ3IDEzNS4yOCA3Mi4yNDIxIDEzNC4xNjcgNzIuMjQyMUgxMDcuMzI2QzEwNi4yMTMgNzIuMjQyMSAxMDUuMzEgNzMuMTQ3IDEwNS4zMSA3NC4yNjMyVjgwLjMyNjNDMTA1LjMxIDgxLjQ0MjUgMTA2LjIxMyA4Mi4zNDc0IDEwNy4zMjYgODIuMzQ3NFpNMTIxLjk0IDEwNi4wOTVIMTQ4Ljc4MUMxNDkuODk0IDEwNi4wOTUgMTUwLjc5NiAxMDUuMTkgMTUwLjc5NiAxMDQuMDc0VjkwLjQzMTZDMTUwLjc5NiA4OS4zMTU0IDE0OS44OTQgODguNDEwNSAxNDguNzgxIDg4LjQxMDVIMTIxLjk0QzEyMC44MjYgODguNDEwNSAxMTkuOTI0IDg5LjMxNTQgMTE5LjkyNCA5MC40MzE2VjEwNC4wNzRDMTE5LjkyNCAxMDUuMTkgMTIwLjgyNiAxMDYuMDk1IDEyMS45NCAxMDYuMDk1Wk0xNDguMTQzIDY0LjE1NzlIMTc0Ljk4NEMxNzYuMDk4IDY0LjE1NzkgMTc3IDYzLjI1MyAxNzcgNjIuMTM2OFY0OC40OTQ3QzE3NyA0Ny4zNzg1IDE3Ni4wOTggNDYuNDczNyAxNzQuOTg0IDQ2LjQ3MzdIMTQ4LjE0M0MxNDcuMDMgNDYuNDczNyAxNDYuMTI4IDQ3LjM3ODUgMTQ2LjEyOCA0OC40OTQ3VjYyLjEzNjhDMTQ2LjEyOCA2My4yNTMgMTQ3LjAzIDY0LjE1NzkgMTQ4LjE0MyA2NC4xNTc5WiIgc3Ryb2tlPSIjMkI1Mjc4Ii8+CjwvZz4KPC9zdmc+Cg==');
  }

  &__btn {
    line-height: 1;
  }

  &-info-param {
    span {
      &:first-child {
        color: #65b9f4;
      }
    }
  }
}

.row {
  margin: 0 0 0 -23px;
}
</style>
