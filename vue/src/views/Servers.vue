<template>
  <div class="page-servers">
    <scrollbar class="page-servers__scroll" @handle-scroll="handleScroll">
      <span
        class="page-servers__btn"
        @click="addNew = true"
        :style="{ left: `${scrollLeft}px`, top: `${scrollTop}px` }"
      >
        <i class="ci-icon ci-plus_circle"></i>
        <span>Добавить сервер</span>
      </span>
      <div class="page-servers__list">
        <LoadSpiner v-show="fetchingServers" text="Получение списка серверов" />
        <ServerTable v-show="showTable && !fetchingServers" @instruction="openInstruction" :servers="servers" />
        <p v-show="!showTable && !fetchingServers" class="page-servers__noserver">
          Нет добавленных серверов демо-панелий
        </p>
      </div>
    </scrollbar>

    <div class="page-servers__new">
      <NewServer v-if="addNew" @addServer="newServer" />
    </div>
    <at-modal v-model="serverModal" class="modal" okText="Читать инструкцию" @on-confirm="openInstruction(serverID)">
      <template v-slot:header><span class="modal-title">Сервер демо-панели добавлен</span></template>
      <p>
        Ознакомьтесь с дальнейшими действиями в
        <span class="clickable" @click="openInstruction(serverID)">Инструкции</span>
      </p>
      <p>Вы также сможете найти ее в таблице серверов на владке Серверы демо-панелей в вашем Профиле</p>
    </at-modal>
    <at-modal
      v-model="InstructionModal"
      class="modal"
      :showConfirmButton="false"
      :showCancelButton="false"
      @on-cancel="buffer = ''"
      :width="600"
    >
      <template v-slot:header><span class="modal-title">Инструкция по настройке сервера демо-панели</span></template>
      <div :class="['server-state', `${selectedServer.state.name}`]">{{ selectedServer.state.value }}</div>
      <div class="server-info">
        <div v-for="(value, key) in selectedServer.info" class="server-info__item" :key="key">
          <p class="label">{{ key }}</p>
          <p class="value">{{ value }}</p>
        </div>
      </div>
      <div class="ssh-wrapper">
        <div class="ssh-wrapper__item">
          <span class="ssh">Приватный SSH-ключ</span>
          <i title="Скопировать" @click="copy('private')" class="btn-copy"></i>
          <a class="clickable" :href="privateURI" download="id_rsa">Скачать</a>
          <span v-show="buffer === 'private'" class="buffer">Ключ скопирован в буффер обмена</span>
        </div>
        <div class="ssh-wrapper__item">
          <span class="ssh">Публичный SSH-ключ</span>
          <i title="Скопировать" @click="copy('public')" class="btn-copy"></i>
          <a class="clickable" :href="publicURI" download="id_rsa.pub">Скачать</a>
          <span v-show="buffer === 'public'" class="buffer">Ключ скопирован в буффер обмена</span>
        </div>
      </div>
      <hr />
      <div class="instruction" v-html="instruction"></div>
    </at-modal>
  </div>
</template>

<script>
import ServerTable from '@/components/servers/ServerTable.vue';
import NewServer from '@/components/servers/NewServer.vue';
import LoadSpiner from '@/components/forms/LoadSpiner';
import { mapGetters } from 'vuex';

export default {
  name: 'servers',
  components: {
    ServerTable,
    NewServer,
    LoadSpiner,
  },
  data: () => ({
    addNew: false,
    serverModal: false,
    InstructionModal: false,
    serverID: null,
    private_key: null,
    public_key: null,
    instruction: null,
    fetchingServers: false,
    buffer: '',
    scrollLeft: 0,
    scrollTop: 0,
    intervalID: null,
  }),
  computed: {
    ...mapGetters({
      servers: 'servers/getServers',
    }),
    showTable() {
      return !!Object.keys(this.$store.getters['servers/getServers']).length;
    },
    getServer() {
      return this.servers.find(server => server.id === this.serverID);
    },
    privateURI() {
      return 'data:application/octet-stream;charset=utf-8,' + encodeURIComponent(this.private_key);
    },
    publicURI() {
      return 'data:application/octet-stream;charset=utf-8,' + encodeURIComponent(this.public_key);
    },
    selectedServer() {
      const selected = this.$store.getters['servers/getServers'].find(server => server.id === this.serverID);
      if (!selected) return { state: '' };
      return {
        info: {
          'Доменное имя': selected.domain_name,
          'Имя пользователя': selected.user,
          'HTTP порт': selected.port_http,
          'IP адерс': selected.ip_address,
          'SSH порт': selected.port_ssh,
          'HTTPS порт': selected.port_https,
        },
        state: selected.state,
      };
    },
  },
  methods: {
    copy(type) {
      let key = type === 'private' ? this.private_key : this.public_key;
      const $el = document.createElement('input');
      document.body.appendChild($el);
      $el.value = key;
      $el.select();
      document.execCommand('copy');
      $el.remove();
      this.buffer = type;
    },
    newServer(id) {
      this.serverID = id;
      this.addNew = false;
      this.serverModal = true;
    },
    async openInstruction(id) {
      this.serverID = id;
      this.serverModal = false;
      this.InstructionModal = true;
      const { data } = await this.$store.dispatch('servers/getInstruction', { id });
      this.private_key = data.private_ssh_key;
      this.public_key = data.public_ssh_key;
      this.instruction = data.instruction;
    },
    handleScroll(vert, horiz) {
      this.scrollLeft = horiz.scrollLeft;
      this.scrollTop = vert.scrollTop;
    },
  },
  async created() {
    this.fetchingServers = true;
    await this.$store.dispatch('servers/getServers');
    this.intervalID = setInterval(async () => {
      await this.$store.dispatch('servers/getServers');
    }, 60000);
    this.fetchingServers = false;
    this.$router.afterEach(() => {
      clearInterval(this.intervalID);
    });
  },
};
</script>

<style lang="scss" scoped>
.page-servers {
  display: grid;
  grid-template: 1fr / 1fr 400px;
  height: 100%;
  &__list {
    height: 100%;
    background: #17212b;
    padding: 60px 20px 0;
    width: 100%;
  }
  &__noserver {
    font-size: 12px;
    color: #a7bed3;
  }
  &__scroll {
    background: #17212b;
  }
  &__btn {
    color: #65b9f4;
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    position: absolute;
    right: 0;
    padding: 30px 20px 10px;
    background: #17212b;
    z-index: 2;
    i {
      font-size: 20px;
    }
  }
  &__new {
    border-left: #0e1621 1px solid;
    background: #17212b;
  }
}

.modal {
  hr {
    border: none;
    border-top: 1px solid #0e1621;
    margin: 15px 0;
  }
  &-title {
    font-size: 16px;
  }
  .instruction {
    margin-bottom: -20px;
  }
  .clickable {
    color: #65b9f4;
    cursor: pointer;
    position: relative;
    &:hover::after {
      content: '';
      position: absolute;
      bottom: -5px;
      left: 0;
      width: 100%;
      height: 1px;
      background: #65b9f4;
    }
  }
  .server-info {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 20px;
    &__item {
      flex: 1 0 170px;
      .label {
        font-size: 0.75rem;
        color: #a7bed3;
        margin-bottom: 5px;
      }
      .value {
        font-size: 1rem;
        color: #fff;
      }
    }
  }
  .error {
    color: #f44336;
  }
  .idle {
    color: #9e9e9e;
  }
  .ready {
    color: #8bc34a;
  }
  .waiting {
    color: #ff9800;
  }
  .server-state {
    font-size: 1rem;
    margin-bottom: 20px;
  }
}

.btn-copy {
  display: inline-block;
  vertical-align: middle;
  cursor: pointer;
  width: 24px;
  height: 24px;
  background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iI0E3QkVEMyI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xNiAxSDRjLTEuMSAwLTIgLjktMiAydjE0aDJWM2gxMlYxem0zIDRIOGMtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxMWMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0wIDE2SDhWN2gxMXYxNHoiLz48L3N2Zz4=');
}

.ssh-wrapper {
  display: flex;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 20px;
  color: #a7bed3;
  &__item {
    flex: 1 0 250px;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
}

.buffer {
  color: #3eba31;
}
</style>